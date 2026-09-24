"""
Solver backends.

All four take the same CanonicalProblem so you can compare them on identical
inputs:

    reference   SLSQP on a smooth reformulation. Ground truth for tests.
    ista        proximal gradient (or FISTA). The textbook baseline.
    analog      the continuous-time network, integrated numerically.
    spiking     the same network with integrate-and-fire neurons.

ista with step 1 is literally forward Euler on the analog dynamics, and the
spiking run time-averages to the same fixed point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .compile import NetworkSpec, compile_network
from .objectives import CanonicalProblem

__all__ = ["Result", "solve_reference", "solve_ista", "solve_analog", "solve_spiking"]


# ---------------------------------------------------------------------------
@dataclass
class Result:
    x: np.ndarray
    y: Optional[np.ndarray] = None          # dual multipliers, if constrained
    objective: float = np.nan
    violation: float = 0.0
    kkt_residual: float = np.nan
    converged: bool = False
    iterations: int = 0
    solver: str = ""
    wall_time_s: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=dict, repr=False)
    stats: Dict[str, Any] = field(default_factory=dict)
    support_tol: float = 1e-9

    @property
    def support(self):
        """Indices we're calling non-zero.

        A spiking readout is never exactly sparse - a neuron that fires once in
        a while still has a tiny rate. Each backend sets support_tol to its own
        noise floor so "sparsity" means the same thing everywhere. If you want
        the strict physical count, look at stats['never_spiked_fraction'].
        """
        return np.flatnonzero(np.abs(self.x) > self.support_tol)

    @property
    def sparsity(self):
        return 1.0 - len(self.support) / max(self.x.size, 1)

    def summary(self) -> str:
        lines = [
            f"solver          : {self.solver}",
            f"objective       : {self.objective:.10g}",
            f"KKT residual    : {self.kkt_residual:.3e}",
            f"max violation   : {self.violation:.3e}",
            f"converged       : {self.converged}  ({self.iterations} iterations)",
            f"sparsity        : {self.sparsity:.1%} "
            f"({len(self.support)}/{self.x.size} active)",
            f"wall time       : {self.wall_time_s * 1e3:.1f} ms",
        ]
        for key in ("total_spikes", "spikes_per_neuron", "active_fraction",
                    "spike_resolution", "synaptic_events", "readout_rate_vs_analog"):
            if key in self.stats:
                v = self.stats[key]
                lines.append(f"{key:<16}: {v:.4g}" if isinstance(v, float)
                             else f"{key:<16}: {v}")
        return "\n".join(lines)


def _finish(res: Result, prob: CanonicalProblem) -> Result:
    """Fill in the quality numbers every backend reports."""
    res.objective = float(prob.objective(res.x))
    res.violation = prob.violation(res.x)
    res.kkt_residual = prob.kkt_residual(res.x, res.y)
    return res


def _prox(v, lam1, lam2, step, nonneg):
    """Prox of step * (lam1 |x| + lam2 x^2), evaluated at v."""
    if nonneg:
        t = np.maximum(v - step * lam1, 0.0)
    else:
        t = np.sign(v) * np.maximum(np.abs(v) - step * lam1, 0.0)
    return t / (1.0 + 2.0 * step * lam2)


def _default_dual_gain(spec: NetworkSpec) -> Optional[float]:
    """Scale the dual step by the largest eigenvalue of A A' so it can't blow up."""
    if not spec.m:
        return None
    return 1.0 / max(1.0, float(np.linalg.eigvalsh(spec.A @ spec.A.T).max()))


# ---------------------------------------------------------------------------
# 1. reference
# ---------------------------------------------------------------------------
def solve_reference(prob: CanonicalProblem, tol: float = 1e-12,
                    max_iter: int = 2000) -> Result:
    """High-accuracy digital solve, for validation only.

    Splits x = p - q with p, q >= 0 so the l1 term becomes linear, then hands
    the smooth constrained problem to SLSQP.
    """
    from scipy.optimize import LinearConstraint, minimize

    t0 = time.perf_counter()
    n = prob.n

    def unpack(z):
        return z[:n] - z[n:]

    def f(z):
        x = unpack(z)
        return (0.5 * x @ prob.Q @ x + prob.c @ x
                + prob.lam1 @ (z[:n] + z[n:]) + prob.lam2 @ (x * x))

    def g(z):
        x = unpack(z)
        gx = prob.Q @ x + prob.c + 2.0 * prob.lam2 * x
        return np.concatenate([gx + prob.lam1, -gx + prob.lam1])

    cons = []
    if prob.A is not None:
        cons.append(LinearConstraint(np.hstack([prob.A, -prob.A]), -np.inf, prob.k))
    if prob.nonneg:
        cons.append(LinearConstraint(np.hstack([-np.eye(n), np.eye(n)]), -np.inf, 0.0))

    out = minimize(f, np.zeros(2 * n), jac=g, bounds=[(0, None)] * (2 * n),
                   constraints=cons, method="SLSQP",
                   options={"maxiter": max_iter, "ftol": tol})

    res = Result(x=unpack(out.x), converged=bool(out.success),
                 iterations=int(out.nit), solver="reference(SLSQP)",
                 wall_time_s=time.perf_counter() - t0)
    if prob.A is not None:
        res.y = _recover_duals(prob, res.x)
    return _finish(res, prob)


def _recover_duals(prob: CanonicalProblem, x, tol=1e-7):
    """Back out y >= 0 from stationarity, using only the active constraints."""
    g = prob.smooth_grad(x)
    sub = np.where(np.abs(x) > tol, prob.lam1 * np.sign(x), 0.0)
    rhs = -(g + sub)
    active = np.flatnonzero(prob.A @ x - prob.k > -1e-6)
    y = np.zeros(prob.m)
    if active.size:
        sol, *_ = np.linalg.lstsq(prob.A[active].T, rhs, rcond=None)
        y[active] = np.maximum(sol, 0.0)
    return y


# ---------------------------------------------------------------------------
# 2. ISTA / FISTA
# ---------------------------------------------------------------------------
def solve_ista(prob: CanonicalProblem, step: Optional[float] = None,
               accelerate: bool = False, max_iter: int = 5000,
               tol: float = 1e-10, record_every: int = 10) -> Result:
    """Proximal gradient descent. Unconstrained problems only.

    With step = 1 and diag(Q) = 1 this is forward Euler on the analog network:
    u = (I - Q)a - c is the membrane potential and each iteration does
    u <- u + u'. The usual 1/lambda_max step is the same iteration on a rescaled
    dictionary - the step limit is a discretisation artefact, not something the
    underlying flow cares about.
    """
    if prob.A is not None:
        raise ValueError("ISTA can't handle linear inequality constraints; "
                         "use backend='analog' or 'spiking'.")

    t0 = time.perf_counter()
    L = float(np.linalg.eigvalsh(prob.Q).max())
    if step is None:
        step = 1.0 / max(L, 1e-12)

    x = np.zeros(prob.n)
    z, t_k = x.copy(), 1.0
    hist = {"iteration": [], "objective": [], "kkt_residual": []}
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        x_new = _prox(z - step * (prob.Q @ z + prob.c),
                      prob.lam1, prob.lam2, step, prob.nonneg)

        if accelerate:
            # Nesterov momentum, standard FISTA schedule
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_k * t_k))
            z = x_new + ((t_k - 1) / t_next) * (x_new - x)
            t_k = t_next
        else:
            z = x_new

        delta = np.linalg.norm(x_new - x)
        x = x_new

        if it % record_every == 0 or it == 1:
            hist["iteration"].append(it)
            hist["objective"].append(float(prob.objective(x)))
            hist["kkt_residual"].append(prob.kkt_residual(x))

        if delta <= tol * max(1.0, np.linalg.norm(x)):
            converged = True
            break

    res = Result(x=x, converged=converged, iterations=it,
                 solver="FISTA" if accelerate else "ISTA",
                 wall_time_s=time.perf_counter() - t0, history=hist,
                 stats={"step_size": step, "lambda_max": L,
                        "unit_step_stable": bool(L < 2.0)})
    return _finish(res, prob)


# ---------------------------------------------------------------------------
# 3. analog network
# ---------------------------------------------------------------------------
def solve_analog(prob: CanonicalProblem, dt: Optional[float] = None,
                 t_max: float = 200.0, dual_gain: Optional[float] = None,
                 tol: float = 1e-10, record_every: int = 20,
                 integrator: str = "euler",
                 spec: Optional[NetworkSpec] = None) -> Result:
    """Integrate the continuous-time network numerically.

        u' = -u + W a + bias - A'v       a = shrink(u, lam1) / nu_f
        w' = beta (A a - k)              v = max(w, 0)

    The dual state w is a signed integrator and v is its rectified readout.
    Rectifying w' instead would make every multiplier monotone, so a constraint
    that was violated once could never go slack again. w is also clamped from
    below so a long feasible stretch can't wind it arbitrarily negative.
    """
    t0 = time.perf_counter()
    spec = spec if spec is not None else compile_network(prob)
    n, m = spec.n, spec.m
    sp = spec.spectrum()

    if dt is None:
        dt = min(0.5, 0.9 * sp["max_euler_step"])
    if dual_gain is None:
        dual_gain = _default_dual_gain(spec)

    u = np.zeros(n)
    w = np.zeros(m)
    v = np.zeros(m)
    w_floor = -10.0 * (1.0 + np.abs(spec.k)) if m else None

    hist = {"time": [], "objective": [], "kkt_residual": [], "violation": []}
    steps = int(np.ceil(t_max / dt))
    converged = False
    i = 0

    def du(u_, v_):
        d = -u_ + spec.W @ spec.activation(u_) + spec.bias
        if m:
            d = d - spec.A.T @ v_
        return d

    for i in range(1, steps + 1):
        if integrator == "rk4":
            k1 = du(u, v)
            k2 = du(u + 0.5 * dt * k1, v)
            k3 = du(u + 0.5 * dt * k2, v)
            k4 = du(u + dt * k3, v)
            delta = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            delta = dt * du(u, v)
        u = u + delta

        a = spec.activation(u)
        if m:
            w = w + dt * dual_gain * (spec.A @ a - spec.k)
            np.maximum(w, w_floor, out=w)
            v = np.maximum(w, 0.0)

        if i % record_every == 0:
            x_o = spec.scaling.to_original(a)
            hist["time"].append(i * dt)
            hist["objective"].append(float(prob.objective(x_o)))
            hist["kkt_residual"].append(prob.kkt_residual(x_o, v if m else None))
            hist["violation"].append(prob.violation(x_o))

        # Settled when the state stops moving and (if constrained) we're feasible.
        if np.linalg.norm(delta) <= tol * dt * max(1.0, np.linalg.norm(u)):
            if not m or np.max(spec.A @ a - spec.k, initial=-1.0) <= 1e-9:
                converged = True
                break

    x = spec.scaling.to_original(spec.activation(u))
    res = Result(x=x, y=(v.copy() if m else None), converged=converged,
                 iterations=i, solver=f"analog-LCA({integrator})",
                 wall_time_s=time.perf_counter() - t0, history=hist,
                 stats={"dt": dt, "sim_time": i * dt, "dual_gain": dual_gain,
                        "normalized": spec.normalized, **sp})
    return _finish(res, prob)


# ---------------------------------------------------------------------------
# 4. spiking network
# ---------------------------------------------------------------------------
def solve_spiking(prob: CanonicalProblem, dt=None, t_max: float = 600.0,
                  spike_resolution="auto", precision: float = 0.01,
                  dual_gain: Optional[float] = None, rate_tau: float = 1.0,
                  readout: str = "filtered", tol: float = 1e-6,
                  record_every: int = 200,
                  spec: Optional[NetworkSpec] = None) -> Result:
    """Run the network with integrate-and-fire neurons.

    One IF neuron per variable:

        mu'  = bias - mu - W_syn * spikes - A'v      soma current
        vm'  = shrink(mu, lam1)                      membrane
        fire when |vm| >= nu_f, then vm -= sign * nu_f

    W_syn = Q - I with a zero diagonal. A neuron's own spikes never reach its
    own soma; that self-inhibition is the reset. Leaving the diagonal in would
    quietly turn a LASSO into an elastic net with lam2 = 1/2.

    spike_resolution (gamma)
        How many spikes make up one unit of activation. The threshold becomes
        nu_f/gamma and the weights W_syn/gamma; the means are unchanged but
        each spike nudges its neighbours less. Readout error is roughly
        (|W| a)_i / gamma - linear in 1/gamma, not 1/sqrt(gamma), because the
        subtractive reset carries the leftover charge forward instead of
        throwing it away (each neuron is a sigma-delta modulator). The error is
        deterministic and not monotone in gamma, so don't try to tune gamma by
        local search; provision from the formula with some margin.

        "auto" does a short analog pre-pass and picks the smallest gamma that
        hits `precision` (relative to the largest coefficient). Halving the
        target error costs 4x the spikes.

    Constraints get one graded-spike neuron each, silent while the constraint
    is slack. If dt * gamma * a_i > 1 a neuron emits several quanta in one
    tick - that's the graded regime, where the payload is an integer count.

    readout
        "filtered"  low-pass of the thresholded soma current (default)
        "rate"      net spike count / (T * gamma) - carries an O(1/T) bias
        "analog"    instantaneous shrink(mu) / nu_f
    """
    t0 = time.perf_counter()
    spec = spec if spec is not None else compile_network(prob)
    info = {}

    if dt is None:
        dt = min(0.2, 0.5 * spec.spectrum()["max_euler_step"])

    if spike_resolution == "auto":
        gamma, required = _auto_spike_resolution(prob, spec, precision)
    else:
        gamma = required = float(spike_resolution)
    info["spike_resolution_required"] = required
    info["resolution_capped"] = bool(required > gamma * (1 + 1e-9))

    if dual_gain is None:
        dual_gain = _default_dual_gain(spec)

    return _run_spiking(prob, spec, dt, t_max, gamma, dual_gain,
                        rate_tau, readout, tol, record_every, t0, info)


def _auto_spike_resolution(prob, spec, precision, gamma_max=5.0e5):
    """Pick gamma from a cheap analog pre-pass.

    Filtering a spike train with unit time constant puts noise of variance
    sum_j W_ij^2 a_j / (2 gamma) into neuron i. Mapped back through the
    variable scaling, we want scale_i * sigma_i <= precision * max|x|.

    If that asks for more than gamma_max we cap it and note the shortfall in
    stats - a badly scaled dictionary is what makes a rate code expensive.
    """
    pre = solve_analog(prob, t_max=60.0, tol=1e-9, record_every=10 ** 9)
    a = np.abs(spec.scaling.to_network(pre.x))
    target = precision * max(float(np.max(np.abs(pre.x), initial=0.0)), 1e-12)
    sigma_max = np.maximum(target / spec.scaling.scale, 1e-12)
    need = (np.abs(spec.W) @ a) / sigma_max
    required = float(np.max(need, initial=1.0))
    return float(np.clip(required, 1.0, gamma_max)), required


def _run_spiking(prob, spec, dt, t_max, gamma, dual_gain, rate_tau,
                 readout, tol, record_every, t0, info):
    n, m = spec.n, spec.m
    W_syn = -spec.W / gamma          # value delivered per spike, zero diagonal
    nu_eff = spec.nu_f / gamma       # threshold per spike quantum

    mu = np.zeros(n)                 # soma current
    vm = np.zeros(n)                 # membrane
    a_filt = np.zeros(n)             # low-passed readout
    net_count = np.zeros(n)          # signed spike count
    spikes = np.zeros(n)
    ever_spiked = np.zeros(n, dtype=bool)
    total_spikes = syn_events = dual_events = 0.0

    w = np.zeros(m)
    v = np.zeros(m)
    w_floor = -10.0 * (1.0 + np.abs(spec.k)) if m else None

    fanout = np.count_nonzero(W_syn, axis=0)
    hist = {"time": [], "objective": [], "kkt_residual": [], "violation": [],
            "spikes": [], "rate_error": []}
    steps = int(np.ceil(t_max / dt))
    converged = False
    prev = None
    i = 0

    for i in range(1, steps + 1):
        # soma: leak toward bias, minus dual feedback, minus incoming spikes
        drive_in = spec.bias if not m else spec.bias - spec.A.T @ v
        mu = mu + dt * (drive_in - mu) - W_syn @ spikes

        # membrane integrates the shrunk current; fire and subtract
        drive = spec.drive(mu)
        vm = vm + dt * drive
        spikes = np.fix(vm / nu_eff)
        if spec.nonneg:
            np.maximum(spikes, 0.0, out=spikes)
        if spikes.any():
            vm = vm - spikes * nu_eff
            mag = np.abs(spikes)
            total_spikes += float(mag.sum())
            syn_events += float(mag @ fanout)
            net_count += spikes
            ever_spiked |= mag > 0

        # filtered readout
        a_filt += (dt / rate_tau) * (drive / spec.nu_f - a_filt)

        # dual layer. Feed it the raw spikes, not the filtered readout: the
        # integrator w already does the averaging, and putting a low-pass in
        # the feedback loop just adds lag and can destabilise it.
        if m:
            w = w + dual_gain * ((spec.A @ spikes) / gamma - spec.k * dt)
            np.maximum(w, w_floor, out=w)
            v = np.maximum(w, 0.0)
            dual_events += float(np.count_nonzero(v))

        if i % record_every == 0:
            t_now = i * dt
            x_o = spec.scaling.to_original(a_filt)
            hist["time"].append(t_now)
            hist["objective"].append(float(prob.objective(x_o)))
            hist["kkt_residual"].append(prob.kkt_residual(x_o, v if m else None))
            hist["violation"].append(prob.violation(x_o))
            hist["spikes"].append(total_spikes)
            hist["rate_error"].append(
                float(np.linalg.norm(net_count / (t_now * gamma) - a_filt)))

            if prev is not None and np.linalg.norm(a_filt - prev) <= tol * max(
                    1.0, np.linalg.norm(a_filt)):
                if not m or np.max(spec.A @ a_filt - spec.k, initial=-1.0) <= 1e-8:
                    converged = True
                    break
            prev = a_filt.copy()

    T = i * dt
    a_rate = net_count / (T * gamma)
    a = {"filtered": a_filt, "rate": a_rate, "analog": spec.activation(mu)}[readout]
    x = spec.scaling.to_original(a)

    # Noise floor on the readout, used to decide what counts as "non-zero".
    sigma = float(np.max(np.abs(spec.W) @ np.abs(a_filt), initial=0.0)
                  / gamma) / float(np.min(spec.nu_f))
    support_tol = max(1e-9, 3.0 * sigma)

    res = Result(
        x=x, y=(v.copy() if m else None), converged=converged, iterations=i,
        solver=f"spiking-LCA[{readout}]", wall_time_s=time.perf_counter() - t0,
        history=hist, support_tol=support_tol,
        stats={
            "dt": dt, "sim_time": T, "spike_resolution": gamma,
            "dual_gain": dual_gain,
            "total_spikes": total_spikes,
            "spikes_per_neuron": total_spikes / max(n, 1),
            "dual_graded_events": dual_events,
            "never_spiked_fraction": float(np.mean(~ever_spiked)),
            "active_fraction": float(np.mean(np.abs(a) > support_tol)),
            "readout_noise_sigma": sigma,
            "readout_rate_vs_analog": float(np.linalg.norm(a_rate - a_filt)),
            "synaptic_events": syn_events,
            "neuron_steps": float(i * (n + m)),
            **spec.spectrum(), **info,
        },
    )
    return _finish(res, prob)