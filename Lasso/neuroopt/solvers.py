"""
Solver backends.

Four of them, deliberately sharing one problem representation so they can be
compared on identical inputs:

``reference``   high-accuracy digital solve (ground truth for tests)
``ista``        proximal gradient / FISTA -- the classical baseline
``analog``      continuous-time LCA / primal-dual flow, integrated numerically
``spiking``     event-driven S-LCA with an integrate-and-fire primal layer and a
                graded-spike constraint layer

``ista`` is exactly unit-step forward Euler on the ``analog`` dynamics, and
``spiking`` time-averages to the same fixed point.  The tests assert this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .compile import NetworkSpec, compile_network
from .objectives import CanonicalProblem

__all__ = ["Result", "solve_reference", "solve_ista", "solve_analog", "solve_spiking"]


# --------------------------------------------------------------------------- #
@dataclass
class Result:
    x: np.ndarray
    y: Optional[np.ndarray] = None          # dual multipliers
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
        """Indices judged non-zero.

        A spiking readout is only *approximately* sparse: a neuron that fires
        even occasionally contributes a tiny positive rate.  ``support_tol`` is
        set by each backend to its own noise floor, so sparsity means the same
        thing across backends.  ``stats['silent_fraction']`` is the stricter,
        purely physical measure: neurons that never emitted a spike.
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
                    "spike_resolution", "synaptic_events",
                    "readout_rate_vs_analog"):
            if key in self.stats:
                v = self.stats[key]
                lines.append(f"{key:<16}: {v:.4g}" if isinstance(v, float)
                             else f"{key:<16}: {v}")
        return "\n".join(lines)


def _finish(res: Result, prob: CanonicalProblem) -> Result:
    res.objective = float(prob.objective(res.x))
    res.violation = prob.violation(res.x)
    res.kkt_residual = prob.kkt_residual(res.x, res.y)
    return res


def _prox(v, lam1, lam2, step, nonneg):
    """prox of ``step*(lam1|x| + lam2 x^2)`` at ``v``."""
    if nonneg:
        t = np.maximum(v - step * lam1, 0.0)
    else:
        t = np.sign(v) * np.maximum(np.abs(v) - step * lam1, 0.0)
    return t / (1.0 + 2.0 * step * lam2)


# --------------------------------------------------------------------------- #
# 1. reference
# --------------------------------------------------------------------------- #
def solve_reference(prob: CanonicalProblem, tol: float = 1e-12,
                    max_iter: int = 2000) -> Result:
    """Ground truth.

    The l1 term is removed by splitting ``x = p - q`` with ``p, q >= 0``, which
    makes the objective smooth and the constraints linear; SLSQP then solves it
    to high accuracy.  Only used for validation, never on hardware.
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
        M = np.hstack([prob.A, -prob.A])
        cons.append(LinearConstraint(M, -np.inf, prob.k))
    if prob.nonneg:
        cons.append(LinearConstraint(np.hstack([-np.eye(n), np.eye(n)]), -np.inf, 0.0))

    z0 = np.zeros(2 * n)
    out = minimize(f, z0, jac=g, bounds=[(0, None)] * (2 * n),
                   constraints=cons, method="SLSQP",
                   options={"maxiter": max_iter, "ftol": tol})

    res = Result(x=unpack(out.x), converged=bool(out.success),
                 iterations=int(out.nit), solver="reference(SLSQP)",
                 wall_time_s=time.perf_counter() - t0)
    if prob.A is not None:
        res.y = _recover_duals(prob, res.x)
    return _finish(res, prob)


def _recover_duals(prob: CanonicalProblem, x, tol=1e-7):
    """Least-squares recovery of ``y >= 0`` from the stationarity condition."""
    g = prob.smooth_grad(x)
    sub = np.where(np.abs(x) > tol, prob.lam1 * np.sign(x), 0.0)
    rhs = -(g + sub)
    active = np.flatnonzero(prob.A @ x - prob.k > -1e-6)
    y = np.zeros(prob.m)
    if active.size:
        sol, *_ = np.linalg.lstsq(prob.A[active].T, rhs, rcond=None)
        y[active] = np.maximum(sol, 0.0)
    return y


# --------------------------------------------------------------------------- #
# 2. ISTA / FISTA
# --------------------------------------------------------------------------- #
def solve_ista(prob: CanonicalProblem, step: Optional[float] = None,
               accelerate: bool = False, max_iter: int = 5000,
               tol: float = 1e-10, record_every: int = 10) -> Result:
    """Proximal gradient descent.  Unconstrained problems only.

    With ``step=1`` and ``diag(Q)=1`` this is *literally* forward Euler on the
    analog LCA dynamics: the auxiliary variable ``u = (I-Q)a - c`` is the
    membrane potential, and the iteration is ``u <- u + 1 * u_dot``.  The usual
    ``step = 1/lambda_max`` is that same iteration applied to a rescaled
    dictionary; the restriction is an artefact of discretisation, not of the
    underlying flow.
    """
    if prob.A is not None:
        raise ValueError("ISTA does not handle linear inequality constraints; "
                         "use backend='analog' or 'spiking'.")
    t0 = time.perf_counter()
    L = float(np.linalg.eigvalsh(prob.Q).max())
    step = step if step is not None else (1.0 / max(L, 1e-12))

    x = np.zeros(prob.n)
    z, t_k = x.copy(), 1.0
    hist = {"iteration": [], "objective": [], "kkt_residual": []}
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        x_new = _prox(z - step * (prob.Q @ z + prob.c), prob.lam1, prob.lam2,
                      step, prob.nonneg)
        if accelerate:
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


# --------------------------------------------------------------------------- #
# 3. analog LCA / primal-dual flow
# --------------------------------------------------------------------------- #
def solve_analog(prob: CanonicalProblem, dt: Optional[float] = None,
                 t_max: float = 200.0, dual_gain: Optional[float] = None,
                 tol: float = 1e-10, record_every: int = 20,
                 integrator: str = "euler",
                 spec: Optional[NetworkSpec] = None) -> Result:
    """Continuous-time dynamics, integrated numerically.

        u_dot = -u + W a + bias - A^T v ,   a = shrink(u, lam1) / nu_f
        w_dot = beta (A a - k) ,            v = [w]_+

    The dual state ``w`` is a *signed* integrator and ``v = [w]_+`` is its
    rectified readout.  Rectifying the derivative instead would make every
    multiplier monotonically non-decreasing, so a constraint that is violated
    once could never become inactive again and complementary slackness would be
    unreachable.  ``w`` is clamped from below (anti-windup) so that long feasible
    stretches cannot wind it arbitrarily negative.
    """
    t0 = time.perf_counter()
    spec = spec if spec is not None else compile_network(prob)
    n, m = spec.n, spec.m
    sp = spec.spectrum()
    if dt is None:
        dt = min(0.5, 0.9 * sp["max_euler_step"])
    if dual_gain is None and m:
        dual_gain = 1.0 / max(1.0, float(np.linalg.eigvalsh(spec.A @ spec.A.T).max()))

    u = np.zeros(n)
    w = np.zeros(m)
    v = np.zeros(m)
    w_floor = -10.0 * (1.0 + np.abs(spec.k)) if m else None

    hist = {"time": [], "objective": [], "kkt_residual": [], "violation": []}
    steps = int(np.ceil(t_max / dt))
    converged = False
    i = 0

    def du(u_, v_):
        a_ = spec.activation(u_)
        d = -u_ + spec.W @ a_ + spec.bias
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
            hist["kkt_residual"].append(
                prob.kkt_residual(x_o, v if m else None))
            hist["violation"].append(prob.violation(x_o))

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


# --------------------------------------------------------------------------- #
# 4. spiking LCA / spiking constrained QP
# --------------------------------------------------------------------------- #
def solve_spiking(prob: CanonicalProblem, dt=None, t_max: float = 600.0,
                  spike_resolution="auto", precision: float = 0.01,
                  dual_gain: Optional[float] = None, rate_tau: float = 1.0,
                  readout: str = "filtered", tol: float = 1e-6,
                  record_every: int = 200,
                  spec: Optional[NetworkSpec] = None) -> Result:
    """Event-driven spiking solver (S-LCA, extended to constrained QP).

    Primal layer -- one integrate-and-fire neuron per decision variable::

        mu_dot  = bias - mu - W_syn sigma - A^T v        (soma current)
        vm_dot  = shrink(mu, lam1)                       (membrane)
        spike when |vm| >= nu_f, then vm -= sign * nu_f  (subtractive reset)

    ``W_syn = Q - I`` has a **zero diagonal**: a neuron is never its own
    presynaptic partner, because its self-inhibition is the subtractive reset.
    Leaving the diagonal in adds ``+a_i`` to the fixed-point equation and
    silently turns a LASSO into an elastic net with ``lam2 = 1/2``.

    Spike resolution
    ----------------
    A spike is a quantum of activation.  With ``gamma`` spikes per unit of
    activation per unit time, the network is simulated with threshold
    ``nu_f/gamma`` and weights ``W_syn/gamma``, which leaves every mean
    unchanged but shrinks the jump each spike makes in the postsynaptic soma
    current.  The residual error in ``mu_i`` is

        sigma_i ~ (|W| a)_i / gamma

    -- **linear** in the quantum size, not the square root of the event count.
    The subtractive reset is why: the leftover membrane charge after a spike is
    bounded by one quantum and is carried forward rather than random-walking,
    which makes each neuron a first-order sigma-delta modulator.  A hard reset
    would discard the remainder and restore a ``1/sqrt(gamma)`` law.

    Measured exponent on a 48x96 sparse coding problem: **-1.010** over two
    decades of gamma (shot noise would give -0.5).

    The error matters because a neuron whose mean current sits *below* the dead
    zone still fires during upward excursions: ``E[shrink(mu)] > shrink(E[mu])``
    by Jensen, so error leaks into the support as spurious small coefficients.

    Two further facts, both measured:

    * The error is **deterministic** -- repeated runs are bit-identical -- and
      quasi-chaotic in ``gamma`` (idle tones).  It is therefore *not monotone*
      in the spike budget, so gamma cannot be tuned by local search.  Provision
      from the law with margin instead.
    * It is caused **entirely by coupling**.  On an orthonormal dictionary
      (``W = 0``) the spiking solver matches the continuous flow to machine
      precision at every gamma.  A neuron's own spikes never reach its own soma
      current (``w_ii = 0``), so all the error is injected by its neighbours.
      A coherent dictionary therefore costs twice: settling time through
      ``lambda_min``, and spike budget through ``|W|``.

    ``spike_resolution="auto"`` runs a short analog pre-pass and picks the
    smallest ``gamma`` meeting ``precision`` -- a target readout error relative
    to the largest coefficient, measured in the *user's* coordinates.  Halving
    the target error costs four times the spikes.

    Constraint layer -- one graded-spike neuron per constraint::

        w_dot = beta (A a - k)  ,   v = [w]_+

    silent whenever the constraint is slack, which is where the event-driven
    advantage comes from.

    When ``dt * gamma * a_i > 1`` a neuron emits several quanta in one tick.
    That is not an artefact: it is the graded-spike regime, where the payload is
    an integer count rather than a single event.

    readout
        ``"filtered"``  low-pass of the thresholded soma current (default)
        ``"rate"``      net spike count / (T * gamma) -- the pure rate code,
                        which additionally carries an O(1/T) transient bias
        ``"analog"``    instantaneous ``shrink(mu)/nu_f``
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

    if dual_gain is None and spec.m:
        dual_gain = 1.0 / max(1.0, float(np.linalg.eigvalsh(spec.A @ spec.A.T).max()))

    return _run_spiking(prob, spec, dt, t_max, gamma, dual_gain,
                        rate_tau, readout, tol, record_every, t0, info)


def _auto_spike_resolution(prob, spec, precision, gamma_max=5.0e5):
    """Pick the value-per-spike from a cheap analog pre-pass.

    A spike train filtered with unit time constant injects shot noise of
    variance ``sum_j W_ij^2 a_j / (2 gamma)`` into neuron i's soma current.
    Mapped back through the variable scaling, the induced error in the user's
    coordinate ``x_i`` is ``scale_i * sigma_i``, so the requirement is

        scale_i * sigma_i <= precision * max_j |x_j| .

    If the resulting gamma exceeds ``gamma_max`` it is capped and the shortfall
    is reported in ``stats``: a badly scaled dictionary makes a rate code
    expensive, which is the quantitative form of the usual advice to normalise
    the atoms to unit norm.
    """
    pre = solve_analog(prob, t_max=60.0, tol=1e-9, record_every=10**9)
    a = np.abs(spec.scaling.to_network(pre.x))
    target = precision * max(float(np.max(np.abs(pre.x), initial=0.0)), 1e-12)
    sigma_max = np.maximum(target / spec.scaling.scale, 1e-12)
    need = (np.abs(spec.W) @ a) / sigma_max
    required = float(np.max(need, initial=1.0))
    return float(np.clip(required, 1.0, gamma_max)), required


def _run_spiking(prob, spec, dt, t_max, gamma, dual_gain, rate_tau,
                 readout, tol, record_every, t0, info):
    n, m = spec.n, spec.m
    W_syn = -spec.W / gamma              # (Q - I), zero diagonal, per-spike value
    nu_eff = spec.nu_f / gamma           # threshold per spike quantum
    ever_spiked = np.zeros(spec.n, dtype=bool)

    mu = np.zeros(n)
    vm = np.zeros(n)
    a_filt = np.zeros(n)                 # low-pass of the thresholded current
    net_count = np.zeros(n)              # cumulative signed spike count
    spikes = np.zeros(n)
    total_spikes = 0.0
    syn_events = 0.0
    dual_events = 0.0

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
        # ---- soma current: leak, bias, inhibitory spikes, dual feedback ----- #
        drive_in = spec.bias if not m else spec.bias - spec.A.T @ v
        mu = mu + dt * (drive_in - mu) - W_syn @ spikes

        # ---- membrane integration and firing (subtractive reset) ------------ #
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

        # ---- filtered readout ----------------------------------------------- #
        a_filt += (dt / rate_tau) * (drive / spec.nu_f - a_filt)

        # ---- constraint layer ------------------------------------------------ #
        if m:
            # Integrate the arriving spike train directly: in the mean this is
            # dt * beta * (A a - k), but it introduces no filter lag inside the
            # primal-dual loop.  Feeding the dual a smoothed readout instead
            # puts a delay in a feedback loop and slows (or destabilises) it --
            # the integrator w is already the averaging element.
            w = w + dual_gain * ((spec.A @ spikes) / gamma - spec.k * dt)
            np.maximum(w, w_floor, out=w)
            v = np.maximum(w, 0.0)
            dual_events += float(np.count_nonzero(v))

        # ---- bookkeeping ------------------------------------------------------ #
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
    a_analog = spec.activation(mu)
    a_rate = net_count / (T * gamma)
    a = {"filtered": a_filt, "rate": a_rate, "analog": a_analog}[readout]
    x = spec.scaling.to_original(a)

    # Readout floor. The subtractive reset makes each neuron a first-order
    # sigma-delta modulator: the residual membrane charge is bounded by one
    # quantum rather than random-walking, so the error scales as 1/gamma.
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
