"""
Solver backends. All take a CanonicalProblem and return a Result.

    reference  SciPy SLSQP, exact, used to check the others
    ista       proximal gradient (accelerate=True gives FISTA)
    analog     the network ODE integrated with Euler
    spiking    the same network with integrate-and-fire neurons
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .compile import compile_network


@dataclass
class Result:
    x: np.ndarray                        # solution, small values set to zero
    x_raw: np.ndarray                    # solution before truncation
    active_threshold: float              # |x| below this counts as inactive
    multipliers: np.ndarray | None = None
    objective: float = np.nan
    violation: float = 0.0
    kkt_residual: float = np.nan
    converged: bool = False
    iterations: int = 0
    solver: str = ""
    wall_time_s: float = 0.0
    history: dict = field(default_factory=dict, repr=False)
    stats: dict = field(default_factory=dict)

    @property
    def active(self):
        return np.flatnonzero(self.x != 0)

    @property
    def n_active(self):
        return len(self.active)

    @property
    def sparsity(self):
        return 1 - self.n_active / max(self.x.size, 1)

    def summary(self):
        lines = [
            f"solver           : {self.solver}",
            f"objective        : {self.objective:.10g}",
            f"KKT residual     : {self.kkt_residual:.3e}",
            f"max violation    : {self.violation:.3e}",
            f"converged        : {self.converged}  ({self.iterations} iterations)",
            f"active threshold : {self.active_threshold:.2e}",
            f"active neurons   : {self.n_active}/{self.x.size}  ({self.sparsity:.0%} sparse)",
            f"wall time        : {self.wall_time_s * 1e3:.1f} ms",
        ]
        for key in ("total_spikes", "spikes_per_neuron", "spikes_per_unit", "synaptic_events", "flops"):
            if key in self.stats:
                lines.append(f"{key:<17}: {self.stats[key]:.4g}")
        return "\n".join(lines)


def _finish(result, problem):
    result.objective = float(problem.objective(result.x))
    result.violation = problem.violation(result.x)
    result.kkt_residual = problem.kkt_residual(result.x, result.multipliers)
    return result


def _make_result(x_raw, problem, active_threshold, **fields):
    x = np.where(np.abs(x_raw) > active_threshold, x_raw, 0.0)
    return _finish(Result(x=x, x_raw=x_raw, active_threshold=active_threshold, **fields), problem)


def _prox(v, l1_weight, l2_weight, step, nonneg):
    if nonneg:
        shrunk = np.maximum(v - step * l1_weight, 0.0)
    else:
        shrunk = np.sign(v) * np.maximum(np.abs(v) - step * l1_weight, 0.0)
    return shrunk / (1 + 2 * step * l2_weight)


def _dual_gain(network):
    if not network.n_constraints:
        return None
    A = network.constraint_matrix
    return 1 / max(1.0, float(np.linalg.eigvalsh(A @ A.T).max()))


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #
def solve_reference(problem, tol=1e-12, max_iter=2000, active_threshold=1e-9):
    """SLSQP on x = plus - minus with plus, minus >= 0, which makes the l1 term linear."""
    from scipy.optimize import LinearConstraint, minimize

    start = time.perf_counter()
    n = problem.size

    def objective(z):
        x = z[:n] - z[n:]
        return (0.5 * x @ problem.gram @ x + problem.linear @ x
                + problem.l1_weight @ (z[:n] + z[n:]) + problem.l2_weight @ x**2)

    def gradient(z):
        x = z[:n] - z[n:]
        g = problem.gram @ x + problem.linear + 2 * problem.l2_weight * x
        return np.concatenate([g + problem.l1_weight, -g + problem.l1_weight])

    constraints = []
    if problem.constraint_matrix is not None:
        A = problem.constraint_matrix
        constraints.append(LinearConstraint(np.hstack([A, -A]), -np.inf, problem.constraint_rhs))
    if problem.nonneg:
        constraints.append(LinearConstraint(np.hstack([-np.eye(n), np.eye(n)]), -np.inf, 0.0))

    out = minimize(objective, np.zeros(2 * n), jac=gradient, bounds=[(0, None)] * (2 * n),
                   constraints=constraints, method="SLSQP",
                   options={"maxiter": max_iter, "ftol": tol})
    x_raw = out.x[:n] - out.x[n:]

    multipliers = _recover_multipliers(problem, x_raw) if problem.constraint_matrix is not None else None
    return _make_result(x_raw, problem, active_threshold, multipliers=multipliers,
                        converged=bool(out.success), iterations=int(out.nit),
                        solver="reference(SLSQP)", wall_time_s=time.perf_counter() - start,
                        stats={"flops": np.nan})


def _recover_multipliers(problem, x, tol=1e-7):
    gradient = problem.smooth_gradient(x)
    subgradient = np.where(np.abs(x) > tol, problem.l1_weight * np.sign(x), 0.0)
    A, rhs = problem.constraint_matrix, problem.constraint_rhs
    active = np.flatnonzero(A @ x - rhs > -1e-6)
    multipliers = np.zeros(problem.n_constraints)
    if active.size:
        solution, *_ = np.linalg.lstsq(A[active].T, -(gradient + subgradient), rcond=None)
        multipliers[active] = np.maximum(solution, 0.0)
    return multipliers


# --------------------------------------------------------------------------- #
# ISTA / FISTA
# --------------------------------------------------------------------------- #
def solve_ista(problem, step=None, accelerate=False, max_iter=5000, tol=1e-10,
               record_every=10, x0=None, active_threshold=1e-9):
    if problem.constraint_matrix is not None:
        raise ValueError("ista doesn't support linear constraints")

    start = time.perf_counter()
    n = problem.size
    lipschitz = float(np.linalg.eigvalsh(problem.gram).max())
    step = step or 1 / max(lipschitz, 1e-12)

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    lookahead = x.copy()
    momentum = 1.0
    history = {"iteration": [], "objective": [], "kkt_residual": []}
    converged = False

    for iteration in range(1, max_iter + 1):
        gradient = problem.gram @ lookahead + problem.linear
        x_new = _prox(lookahead - step * gradient, problem.l1_weight, problem.l2_weight, step, problem.nonneg)

        if accelerate:
            momentum_new = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            lookahead = x_new + (momentum - 1) / momentum_new * (x_new - x)
            momentum = momentum_new
        else:
            lookahead = x_new

        change = np.linalg.norm(x_new - x)
        x = x_new

        if iteration == 1 or iteration % record_every == 0:
            history["iteration"].append(iteration)
            history["objective"].append(float(problem.objective(x)))
            history["kkt_residual"].append(problem.kkt_residual(x))

        if change <= tol * max(1.0, np.linalg.norm(x)):
            converged = True
            break

    # one matrix-vector product per iteration: 2 n^2 flops
    flops = iteration * (2 * n * n + 8 * n)
    return _make_result(x, problem, active_threshold, converged=converged, iterations=iteration,
                        solver="FISTA" if accelerate else "ISTA",
                        wall_time_s=time.perf_counter() - start, history=history,
                        stats={"step": step, "lambda_max": lipschitz, "flops": flops})


# --------------------------------------------------------------------------- #
# analog
# --------------------------------------------------------------------------- #
def solve_analog(problem, dt=None, t_max=600.0, dual_gain=None, tol=1e-10,
                 record_every=20, x0=None, network=None, active_threshold=1e-9):
    start = time.perf_counter()
    network = network or compile_network(problem)
    n, m = network.size, network.n_constraints
    spectrum = network.spectrum()

    dt = dt or min(0.5, 0.9 * spectrum["max_euler_step"])
    if dual_gain is None:
        dual_gain = _dual_gain(network)

    membrane = np.zeros(n)
    if x0 is not None:
        # membrane = activation * threshold + dead_zone * sign gives activation back after shrink
        a0 = network.scaling.to_network(np.asarray(x0, dtype=float))
        membrane = a0 * network.threshold + network.dead_zone * np.sign(a0)

    dual_state = np.zeros(m)
    multipliers = np.zeros(m)
    dual_floor = -10 * (1 + np.abs(network.constraint_rhs)) if m else None

    history = {"time": [], "objective": [], "kkt_residual": [], "violation": []}
    n_steps = int(np.ceil(t_max / dt))
    converged = False
    step_index = 0

    for step_index in range(1, n_steps + 1):
        activation = network.activation(membrane)
        change = dt * (-membrane + network.weights @ activation + network.input_current)
        if m:
            change -= dt * network.constraint_matrix.T @ multipliers
        membrane = membrane + change

        activation = network.activation(membrane)
        if m:
            dual_state = dual_state + dt * dual_gain * (network.constraint_matrix @ activation - network.constraint_rhs)
            np.maximum(dual_state, dual_floor, out=dual_state)
            multipliers = np.maximum(dual_state, 0.0)

        if step_index % record_every == 0:
            x_now = network.scaling.to_user(activation)
            history["time"].append(step_index * dt)
            history["objective"].append(float(problem.objective(x_now)))
            history["kkt_residual"].append(problem.kkt_residual(x_now, multipliers if m else None))
            history["violation"].append(problem.violation(x_now))

        if tol is not None and np.linalg.norm(change) <= tol * dt * max(1.0, np.linalg.norm(membrane)):
            if not m or np.max(network.constraint_matrix @ activation - network.constraint_rhs, initial=-1.0) <= 1e-9:
                converged = True
                break

    flops = step_index * (2 * n * n + 10 * n + 4 * m * n)
    x_raw = network.scaling.to_user(network.activation(membrane))
    return _make_result(x_raw, problem, active_threshold,
                        multipliers=multipliers.copy() if m else None,
                        converged=converged, iterations=step_index, solver="analog-LCA",
                        wall_time_s=time.perf_counter() - start, history=history,
                        stats={"dt": dt, "sim_time": step_index * dt, "dual_gain": dual_gain,
                               "rescaled": network.rescaled, "flops": flops, **spectrum})


# --------------------------------------------------------------------------- #
# spiking
# --------------------------------------------------------------------------- #
def solve_spiking(problem, dt=None, t_max=600.0, spikes_per_unit="auto", precision=0.01,
                  dual_gain=None, filter_tau=1.0, readout="filtered", tol=1e-6,
                  record_every=200, x0=None, network=None, active_threshold="auto"):
    """
    Integrate-and-fire network with subtractive reset.

    spikes_per_unit: how many spikes represent one unit of activation. "auto"
    runs a short analog pass and picks the smallest value meeting `precision`.

    readout: "filtered" (low-passed soma current), "rate" (spike count / time),
    or "analog" (instantaneous).

    active_threshold: "auto" uses 3x the estimated spike-noise floor.

    tol=None runs the full t_max with no early stop (for convergence plots).
    """
    start = time.perf_counter()
    network = network or compile_network(problem)

    dt = dt or min(0.2, 0.5 * network.spectrum()["max_euler_step"])
    if spikes_per_unit == "auto":
        spikes_per_unit, required = _choose_spikes_per_unit(problem, network, precision)
    else:
        spikes_per_unit = required = float(spikes_per_unit)
    if dual_gain is None:
        dual_gain = _dual_gain(network)

    extra = {"spikes_per_unit_required": required,
             "resolution_capped": required > spikes_per_unit * (1 + 1e-9)}
    return _run_spiking(problem, network, dt, t_max, spikes_per_unit, dual_gain, filter_tau,
                        readout, tol, record_every, x0, active_threshold, start, extra)


def _choose_spikes_per_unit(problem, network, precision, cap=5e5):
    # readout noise on neuron i ~ (|weights| @ activation)_i / spikes_per_unit
    pre = solve_analog(problem, t_max=60.0, tol=1e-9, record_every=10**9)
    activation = np.abs(network.scaling.to_network(pre.x_raw))
    target = precision * max(float(np.abs(pre.x_raw).max(initial=0.0)), 1e-12)
    allowed_noise = np.maximum(target / network.scaling.factor, 1e-12)
    required = float(np.max(np.abs(network.weights) @ activation / allowed_noise, initial=1.0))
    if network.n_constraints:
        # constraint neurons also see spike quantisation through the constraint matrix
        A = network.constraint_matrix
        constraint_noise = np.abs(A) @ activation
        required = max(required, float(np.max(constraint_noise, initial=0.0) / target))
    return float(np.clip(required, 1.0, cap)), required


def _run_spiking(problem, network, dt, t_max, spikes_per_unit, dual_gain, filter_tau,
                 readout, tol, record_every, x0, active_threshold, start, extra):
    n, m = network.size, network.n_constraints
    synapse_weights = -network.weights / spikes_per_unit
    fire_level = network.threshold / spikes_per_unit

    soma_current = np.zeros(n)
    membrane = np.zeros(n)
    filtered_output = np.zeros(n)
    spike_count = np.zeros(n)
    spikes = np.zeros(n)
    ever_spiked = np.zeros(n, dtype=bool)
    total_spikes = synaptic_events = dual_events = 0.0

    if x0 is not None:
        a0 = network.scaling.to_network(np.asarray(x0, dtype=float))
        soma_current = a0 * network.threshold + network.dead_zone * np.sign(a0)
        filtered_output = a0.copy()

    dual_state = np.zeros(m)
    multipliers = np.zeros(m)
    dual_floor = -10 * (1 + np.abs(network.constraint_rhs)) if m else None

    fan_out = np.count_nonzero(synapse_weights, axis=0)
    history = {key: [] for key in ("time", "objective", "kkt_residual", "violation", "spikes")}
    n_steps = int(np.ceil(t_max / dt))
    converged = False
    previous_output = None
    settled_checks = 0
    tick = 0

    for tick in range(1, n_steps + 1):
        drive = network.input_current - network.constraint_matrix.T @ multipliers if m else network.input_current
        soma_current = soma_current + dt * (drive - soma_current) - synapse_weights @ spikes

        shrunk = network.shrink(soma_current)
        membrane = membrane + dt * shrunk
        spikes = np.fix(membrane / fire_level)
        if network.rectified:
            np.maximum(spikes, 0.0, out=spikes)
        if spikes.any():
            membrane -= spikes * fire_level
            magnitude = np.abs(spikes)
            total_spikes += magnitude.sum()
            synaptic_events += magnitude @ fan_out
            spike_count += spikes
            ever_spiked |= magnitude > 0

        filtered_output += dt / filter_tau * (shrunk / network.threshold - filtered_output)

        if m:
            dual_state = dual_state + dual_gain * (network.constraint_matrix @ spikes / spikes_per_unit
                                                   - network.constraint_rhs * dt)
            np.maximum(dual_state, dual_floor, out=dual_state)
            multipliers = np.maximum(dual_state, 0.0)
            dual_events += np.count_nonzero(multipliers)

        if tick % record_every == 0:
            x_now = network.scaling.to_user(filtered_output)
            history["time"].append(tick * dt)
            history["objective"].append(float(problem.objective(x_now)))
            history["kkt_residual"].append(problem.kkt_residual(x_now, multipliers if m else None))
            history["violation"].append(problem.violation(x_now))
            history["spikes"].append(total_spikes)

            if tol is not None and previous_output is not None:
                # can't settle tighter than the spike noise floor, so stop there
                noise_now = float(np.max(np.abs(network.weights) @ np.abs(filtered_output), initial=0.0)) / spikes_per_unit
                settle_tol = max(tol, noise_now)
                settled = np.linalg.norm(filtered_output - previous_output) <= settle_tol * max(1.0, np.linalg.norm(filtered_output))
                feasible = not m or np.max(network.constraint_matrix @ filtered_output - network.constraint_rhs, initial=-1.0) <= 1e-8
                settled_checks = settled_checks + 1 if (settled and feasible) else 0
                if settled_checks >= 2:
                    converged = True
                    break
            previous_output = filtered_output.copy()

    sim_time = tick * dt
    rate_output = spike_count / (sim_time * spikes_per_unit)
    activation = {"filtered": filtered_output, "rate": rate_output, "analog": network.activation(soma_current)}[readout]

    noise_floor = float(np.max(np.abs(network.weights) @ np.abs(filtered_output), initial=0.0)) / spikes_per_unit / float(network.threshold.min())
    if active_threshold == "auto":
        active_threshold = max(1e-9, 3 * noise_floor)

    # software cost of the simulation: dense matvec per tick
    flops = tick * (2 * n * n + 12 * n + 4 * m * n)

    return _make_result(
        network.scaling.to_user(activation), problem, active_threshold,
        multipliers=multipliers.copy() if m else None,
        converged=converged, iterations=tick,
        solver=f"spiking-LCA[{readout}]",
        wall_time_s=time.perf_counter() - start, history=history,
        stats={
            "dt": dt,
            "sim_time": sim_time,
            "spikes_per_unit": spikes_per_unit,
            "dual_gain": dual_gain,
            "total_spikes": float(total_spikes),
            "spikes_per_neuron": float(total_spikes) / max(n, 1),
            "synaptic_events": float(synaptic_events),
            "neuron_updates": float(tick * (n + m)),
            "dual_events": float(dual_events),
            "never_spiked_fraction": float(np.mean(~ever_spiked)),
            "noise_floor": noise_floor,
            "rate_vs_filtered": float(np.linalg.norm(rate_output - filtered_output)),
            "flops": flops,
            **network.spectrum(),
            **extra,
        },
    )
