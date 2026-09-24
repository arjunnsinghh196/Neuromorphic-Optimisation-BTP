"""
Public entry points.

    from neuroopt import lasso, solve, describe

    prob = lasso(Phi, s, lam=0.1)
    print(describe(prob))                  # what network does this become?
    res  = solve(prob, backend="spiking")
    print(res.summary())

The backend is a choice about *how* to run the dynamics, not *what* is being
solved. Every backend sees the same canonical problem.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .compile import NetworkSpec, compile_network
from .objectives import (
    Box,
    CanonicalProblem,
    L1,
    L2Squared,
    LeastSquares,
    LinearInequality,
    NonNegative,
    Problem,
    Quadratic,
)
from .solvers import Result, solve_analog, solve_ista, solve_reference, solve_spiking

__all__ = [
    "solve", "describe", "network_for",
    "lasso", "elastic_net", "ridge", "nnls", "quadratic_program", "sparse_qp",
]

_BACKENDS = {
    "spiking": solve_spiking,
    "analog": solve_analog,
    "ista": solve_ista,
    "fista": lambda p, **kw: solve_ista(p, accelerate=True, **kw),
    "reference": solve_reference,
}


def _canon(problem: Union[Problem, CanonicalProblem]) -> CanonicalProblem:
    return problem if isinstance(problem, CanonicalProblem) else problem.canonicalize()


def solve(problem, backend: str = "spiking", **kwargs) -> Result:
    """Solve a problem with one of the backends.

    backend:
        "spiking"     integrate-and-fire network (default)
        "analog"      same dynamics in continuous time
        "ista"        proximal gradient; "fista" for the accelerated version
        "reference"   slow, accurate digital solve, for checking the others

    Extra keyword arguments go straight to the backend.
    """
    prob = _canon(problem)
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; pick one of {sorted(_BACKENDS)}")
    if backend in ("ista", "fista") and prob.A is not None:
        raise ValueError("constrained problems need backend='spiking' or 'analog'")
    return _BACKENDS[backend](prob, **kwargs)


def network_for(problem, **kw) -> NetworkSpec:
    """Compile without solving - handy for looking at weights before you deploy."""
    return compile_network(_canon(problem), **kw)


def describe(problem) -> str:
    """Plain-text report of the neuron model this objective compiles to."""
    prob = _canon(problem)
    spec = compile_network(prob)
    sp = spec.spectrum()

    has_l1 = bool(np.any(prob.lam1 > 0))
    has_l2 = bool(np.any(prob.lam2 > 0))

    if prob.constrained:
        neuron = "two-layer: IF primal + graded-spike constraint neurons"
        fit = "good - constraint neurons only spike while active"
    elif has_l1 and has_l2:
        neuron = "IF with dead zone and reduced gain"
        fit = "good"
    elif has_l1:
        neuron = "integrate-and-fire with dead zone"
        fit = "excellent - sparse solution means sparse spikes"
    elif has_l2:
        neuron = "linear integrator with extra leak"
        fit = "poor - dense solution, every neuron fires constantly"
    else:
        neuron = "linear leaky integrator"
        fit = "poor - no sparsity, spike count ~ N*T"

    gain = 1.0 / (2.0 * prob.lam2 + 1.0)

    def show(v):
        """Print a scalar if the vector is constant, otherwise its range."""
        if np.allclose(v, v[0]):
            return f"{v.min():.4g}"
        return f"[{v.min():.4g}, {v.max():.4g}]"

    lines = [
        "objective -> neural architecture",
        "-" * 52,
        f"decision variables      : {prob.n}",
        f"linear constraints      : {prob.m}",
        f"neuron model            : {neuron}",
        f"recurrent weights       : W = I - Q  (zero diagonal, "
        f"{np.count_nonzero(spec.W)} synapses)",
        f"bias current            : -c",
        f"dead zone (lam1)        : {show(prob.lam1)}",
        f"firing threshold nu_f   : {show(spec.nu_f)}   (= 2*lam2 + 1)",
        f"gain above threshold    : {show(gain)}",
        f"non-negative neurons    : {prob.nonneg}",
        f"variable rescaling      : "
        f"{'applied (diag(Q) was not 1)' if spec.normalized else 'not needed'}",
        "",
        "convergence",
        "-" * 52,
        f"lambda_min              : {sp['lambda_min']:.4g}",
        f"lambda_max              : {sp['lambda_max']:.4g}",
        f"condition number        : {sp['condition_number']:.4g}",
        f"max stable Euler step   : {sp['max_euler_step']:.4g}",
        f"unit-step Euler (=ISTA) : "
        f"{'stable' if sp['lambda_max'] < 2 else 'UNSTABLE, needs rescaling'}",
        f"hardware fit            : {fit}",
    ]

    if sp["lambda_min"] <= 1e-10:
        lines += [
            "",
            "note: Q is rank deficient. That's normal for an overcomplete",
            "      dictionary - convergence is then set by the smallest eigenvalue",
            "      of Q on the active set, not by lambda_min(Q).",
        ]
    elif sp["condition_number"] > 1e3:
        lines += [
            "",
            "note: Q is badly conditioned. The continuous-time network doesn't",
            "      pay the kappa penalty of discrete solvers, but settling time is",
            "      still ~1/lambda_min. Adding an l2 term shifts every eigenvalue",
            "      up by 2*lam2 and helps.",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ready-made problems
# ---------------------------------------------------------------------------
def _with_constraints(p: Problem, constraints, n: int, nonneg: bool) -> Problem:
    cons = list(constraints) + ([NonNegative(n)] if nonneg else [])
    return p.subject_to(*cons) if cons else p


def lasso(Phi, s, lam, nonneg: bool = False, constraints=()) -> Problem:
    """min 1/2 ||s - Phi a||^2 + lam ||a||_1"""
    n = np.asarray(Phi).shape[1]
    p = Problem([LeastSquares(Phi, s), L1(lam, n=n)])
    return _with_constraints(p, constraints, n, nonneg)


def elastic_net(Phi, s, lam1, lam2, nonneg: bool = False, constraints=()) -> Problem:
    """min 1/2 ||s - Phi a||^2 + lam1 ||a||_1 + lam2 ||a||_2^2

    Squared l2. The unsquared one is a different problem with a different neuron.
    """
    n = np.asarray(Phi).shape[1]
    p = Problem([LeastSquares(Phi, s), L1(lam1, n=n), L2Squared(lam2, n=n)])
    return _with_constraints(p, constraints, n, nonneg)


def ridge(Phi, s, lam) -> Problem:
    """min 1/2 ||s - Phi a||^2 + lam ||a||_2^2

    Here mostly for contrast: easy to solve, but the solution is dense so every
    neuron spikes. Not a good fit for spiking hardware.
    """
    n = np.asarray(Phi).shape[1]
    return Problem([LeastSquares(Phi, s), L2Squared(lam, n=n)])


def nnls(Phi, s, lam=0.0) -> Problem:
    """Non-negative least squares. The rectifier is the neuron, not a constraint."""
    return lasso(Phi, s, lam, nonneg=True)


def quadratic_program(Q, c, A=None, k=None, lo=None, hi=None,
                      nonneg: bool = False) -> Problem:
    """min 1/2 x'Qx + c'x   s.t.  Ax <= k,  lo <= x <= hi"""
    n = np.asarray(Q).shape[0]
    p = Problem([Quadratic(Q, c)])
    cons = []
    if A is not None:
        cons.append(LinearInequality(A, k))
    if lo is not None or hi is not None:
        cons.append(Box(lo, hi, n=n))
    if nonneg:
        cons.append(NonNegative(n))
    return p.subject_to(*cons) if cons else p


def sparse_qp(Q, c, lam1, A=None, k=None, lam2=0.0) -> Problem:
    """min 1/2 x'Qx + c'x + lam1||x||_1 + lam2||x||_2^2   s.t.  Ax <= k

    The most general thing this library maps directly.
    """
    n = np.asarray(Q).shape[0]
    terms = [Quadratic(Q, c), L1(lam1, n=n)]
    if np.any(np.asarray(lam2) > 0):
        terms.append(L2Squared(lam2, n=n))
    p = Problem(terms)
    return p.subject_to(LinearInequality(A, k)) if A is not None else p