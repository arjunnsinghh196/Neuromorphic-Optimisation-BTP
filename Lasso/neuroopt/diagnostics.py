"""Cross-backend comparison and the measurements the benchmarks usually omit."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .compile import compile_network
from .objectives import CanonicalProblem, Problem

__all__ = ["compare", "conditioning_report", "spike_report"]


def _canon(problem):
    return problem if isinstance(problem, CanonicalProblem) else problem.canonicalize()


def compare(problem, backends: Sequence[str] = ("reference", "fista", "analog", "spiking"),
            options: Dict[str, dict] | None = None, verbose: bool = True):
    """Run several backends on the same problem and tabulate the result."""
    from .api import solve

    prob = _canon(problem)
    options = options or {}
    if prob.A is not None:
        backends = [b for b in backends if b not in ("ista", "fista")]

    results = {}
    ref = None
    for b in backends:
        res = solve(prob, backend=b, **options.get(b, {}))
        results[b] = res
        if b == "reference":
            ref = res

    if verbose:
        head = (f"{'backend':<22}{'objective':>16}{'KKT':>11}{'viol':>10}"
                f"{'|x-ref|':>11}{'nnz':>6}{'spikes':>11}")
        print(head)
        print("-" * len(head))
        for b, r in results.items():
            dx = (np.linalg.norm(r.x - ref.x) if ref is not None else np.nan)
            spk = r.stats.get("total_spikes", float("nan"))
            print(f"{r.solver:<22}{r.objective:>16.10g}{r.kkt_residual:>11.2e}"
                  f"{r.violation:>10.2e}{dx:>11.2e}{len(r.support):>6}"
                  f"{spk:>11.6g}")
    return results


def conditioning_report(problem, x=None) -> dict:
    """Spectrum of the effective Gram matrix, and what it costs each solver.

    Discrete solvers pay ``kappa * log(1/eps)`` iterations; continuous dynamics
    pay ``(1/lambda_min) * log(1/eps)`` in physical time.  Neuromorphic hardware
    removes the dependence on ``lambda_max``, not the dependence on
    ``lambda_min`` -- a flat direction is still slow, it is just no longer slow
    *because* some other direction is steep.
    """
    prob = _canon(problem)
    spec = compile_network(prob)
    G = spec.Q + np.diag(spec.nu_f - 1.0)   # Q + 2*lam2 I
    ev = np.linalg.eigvalsh(G)
    if x is None and np.any(prob.lam1 > 0):
        from .api import solve
        x = solve(prob, backend="analog", t_max=400).x
    lo, hi = float(ev.min()), float(ev.max())
    kappa = hi / lo if lo > 1e-14 else np.inf
    eps = 1e-8
    # For an overcomplete dictionary lambda_min(G) is ~0 by construction and says
    # nothing useful.  Once the active set has settled, the rate is governed by
    # the smallest eigenvalue of G restricted to the support -- which is what a
    # sparse problem actually pays.
    restricted = None
    if x is not None:
        gamma_idx = np.flatnonzero(np.abs(x) > 1e-8)
        if gamma_idx.size:
            restricted = float(np.linalg.eigvalsh(G[np.ix_(gamma_idx, gamma_idx)]).min())

    return {
        "lambda_min": lo,
        "support_size": (0 if x is None else int(np.sum(np.abs(x) > 1e-8))),
        "lambda_min_on_support": restricted,
        "restricted_settling_time_to_1e-8": (
            np.log(1 / eps) / restricted if restricted and restricted > 1e-14 else np.inf),
        "lambda_max": hi,
        "condition_number": kappa,
        "ista_step_limit": 2.0 / hi,
        "ista_iterations_to_1e-8": (kappa * np.log(1 / eps)) if np.isfinite(kappa) else np.inf,
        "continuous_settling_time_to_1e-8": (np.log(1 / eps) / lo) if lo > 1e-14 else np.inf,
        "l2_shift_applied": float(np.max(spec.nu_f - 1.0)),
        "advice": (
            "well conditioned" if kappa < 50 else
            "rank-deficient Gram (expected for an overcomplete dictionary): read "
            "lambda_min_on_support, not lambda_min" if lo <= 1e-10 else
            "ill conditioned: add an l2 term (shifts every eigenvalue by 2*lam2, "
            "reducing kappa monotonically) or decorrelate the dictionary"
        ),
    }


def spike_report(result) -> dict:
    """Spike economy of a spiking run, including the O(1/T) rate-readout floor."""
    s = result.stats
    if "total_spikes" not in s:
        raise ValueError("not a spiking result")
    T = s["sim_time"]
    return {
        "total_spikes": s["total_spikes"],
        "spikes_per_neuron": s["spikes_per_neuron"],
        "spikes_per_active_neuron": (
            s["total_spikes"] / max(1.0, s["active_fraction"] * result.x.size)),
        "active_fraction": s["active_fraction"],
        "dual_graded_events": s.get("dual_graded_events", 0.0),
        "synaptic_events": s.get("synaptic_events", 0.0),
        "sim_time": T,
        "rate_readout_error": s["readout_rate_vs_analog"],
        "rate_readout_bound_O(1/T)": 1.0 / T,
    }
