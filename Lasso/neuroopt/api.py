"""
Public entry points.

    from neuroopt import lasso, solve, describe

    prob = lasso(Phi, s, lam=0.1)
    print(describe(prob))          # which neuron, which weights, which threshold
    res  = solve(prob, backend="spiking")
    print(res.summary())

Every problem is specified as an objective plus constraints; the backend is a
choice about *how* to realise the same dynamical system, not about what is being
solved.
"""

from __future__ import annotations

from typing import Optional, Union

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
    """Solve ``problem`` with the chosen backend.

    backend
        ``"spiking"``   integrate-and-fire network (default; the point of this library)
        ``"analog"``    the same dynamics in continuous time, numerically integrated
        ``"ista"`` / ``"fista"``   classical proximal gradient baseline
        ``"reference"`` high-accuracy digital solve, for validation
    """
    prob = _canon(problem)
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(_BACKENDS)}")
    fn = _BACKENDS[backend]
    if backend in ("ista", "fista") and prob.A is not None:
        raise ValueError("constrained problems need backend='spiking' or 'analog'")
    return fn(prob, **kwargs)


def network_for(problem, **kw) -> NetworkSpec:
    """Compile without solving -- useful for inspecting weights before mapping."""
    return compile_network(_canon(problem), **kw)


def describe(problem) -> str:
    """Report the neuron model this objective compiles to.

    This is the library's actual purpose: objective in, neuron model out.
    """
    prob = _canon(problem)
    spec = compile_network(prob)
    sp = spec.spectrum()

    has_l1 = bool(np.any(prob.lam1 > 0))
    has_l2 = bool(np.any(prob.lam2 > 0))
    if prob.constrained:
        neuron = "two-layer: IF primal + graded-spike constraint neurons"
        fit = "good -- spikes only while constraints are active"
    elif has_l1 and has_l2:
        neuron = "modified IF (dead zone + reduced gain)"
        fit = "good"
    elif has_l1:
        neuron = "integrate-and-fire with dead zone"
        fit = "excellent -- sparse solution gives sparse spikes"
    elif has_l2:
        neuron = "linear integrator with increased leak"
        fit = "poor -- dense solution, every neuron spikes continuously"
    else:
        neuron = "linear leaky integrator"
        fit = "poor -- no sparsity, spike count ~ N*T"

    lam1 = prob.lam1
    lam2 = prob.lam2
    slope = 1.0 / (2.0 * lam2 + 1.0)

    def rng(v):
        return f"{v.min():.4g}" if np.allclose(v, v[0]) else f"[{v.min():.4g}, {v.max():.4g}]"

    lines = [
        "objective -> neural architecture",
        "-" * 52,
        f"decision variables      : {prob.n}",
        f"linear constraints      : {prob.m}",
        f"neuron model            : {neuron}",
        f"recurrent weights       : W = I - Q  (zero diagonal, "
        f"{np.count_nonzero(spec.W)} synapses)",
        f"bias current            : -c",
        f"dead zone (lam1)        : {rng(lam1)}",
        f"firing threshold nu_f   : {rng(spec.nu_f)}   (= 2*lam2 + 1)",
        f"gain above threshold    : {rng(slope)}",
        f"non-negative neurons    : {prob.nonneg}",
        f"variable rescaling      : {'applied (diag(Q) was not 1)' if spec.normalized else 'not needed'}",
        "",
        "convergence",
        "-" * 52,
        f"lambda_min              : {sp['lambda_min']:.4g}",
        f"lambda_max              : {sp['lambda_max']:.4g}",
        f"condition number        : {sp['condition_number']:.4g}",
        f"max stable Euler step   : {sp['max_euler_step']:.4g}",
        f"unit-step Euler (=ISTA) : {'stable' if sp['lambda_max'] < 2 else 'UNSTABLE, needs rescaling'}",
        f"hardware fit            : {fit}",
    ]
    if sp["lambda_min"] <= 1e-10 and prob.n > prob.Q.shape[0] - 1:
        lines += ["", "note: Q is rank deficient, which is normal for an overcomplete",
                  "      dictionary. Convergence is then governed by the smallest",
                  "      eigenvalue of Q restricted to the active set, not by",
                  "      lambda_min(Q); see conditioning_report(problem)."]
    elif sp["condition_number"] > 1e3:
        lines += ["", "note: ill-conditioned Gram matrix. Continuous-time dynamics remove",
                  "      the kappa dependence of discrete solvers but not the",
                  "      1/lambda_min settling time. An l2 term shifts every eigenvalue",
                  "      by 2*lam2 and reduces kappa monotonically."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# standard problem constructors
# --------------------------------------------------------------------------- #
def lasso(Phi, s, lam, nonneg: bool = False, constraints=()) -> Problem:
    """``min 1/2 ||s - Phi a||^2 + lam ||a||_1``  (optionally constrained)."""
    n = np.asarray(Phi).shape[1]
    p = Problem([LeastSquares(Phi, s), L1(lam, n=n)])
    cons = list(constraints) + ([NonNegative(n)] if nonneg else [])
    return p.subject_to(*cons) if cons else p


def elastic_net(Phi, s, lam1, lam2, nonneg: bool = False, constraints=()) -> Problem:
    """``min 1/2 ||s - Phi a||^2 + lam1 ||a||_1 + lam2 ||a||_2^2``.

    Note the *squared* l2 norm: the unsquared norm has a block-thresholding
    proximal operator and a different neuron.
    """
    n = np.asarray(Phi).shape[1]
    p = Problem([LeastSquares(Phi, s), L1(lam1, n=n), L2Squared(lam2, n=n)])
    cons = list(constraints) + ([NonNegative(n)] if nonneg else [])
    return p.subject_to(*cons) if cons else p


def ridge(Phi, s, lam) -> Problem:
    """``min 1/2 ||s - Phi a||^2 + lam ||a||_2^2``.  Included for contrast: it is
    convex and easy but produces a dense solution, so every neuron spikes."""
    n = np.asarray(Phi).shape[1]
    return Problem([LeastSquares(Phi, s), L2Squared(lam, n=n)])


def nnls(Phi, s, lam=0.0) -> Problem:
    """Non-negative least squares; the rectifier is the neuron, not a constraint layer."""
    return lasso(Phi, s, lam, nonneg=True)


def quadratic_program(Q, c, A=None, k=None, lo=None, hi=None,
                      nonneg: bool = False) -> Problem:
    """``min 1/2 x'Qx + c'x  s.t.  Ax <= k,  lo <= x <= hi``."""
    p = Problem([Quadratic(Q, c)])
    cons = []
    if A is not None:
        cons.append(LinearInequality(A, k))
    if lo is not None or hi is not None:
        cons.append(Box(lo, hi, n=np.asarray(Q).shape[0]))
    if nonneg:
        cons.append(NonNegative(np.asarray(Q).shape[0]))
    return p.subject_to(*cons) if cons else p


def sparse_qp(Q, c, lam1, A=None, k=None, lam2=0.0) -> Problem:
    """``min 1/2 x'Qx + c'x + lam1||x||_1 + lam2||x||_2^2  s.t. Ax <= k``.

    The widest form this library maps directly: l1 sets the dead zone, l2 sets the
    firing threshold, Q sets the recurrent weights, A sets the constraint layer.
    """
    n = np.asarray(Q).shape[0]
    terms = [Quadratic(Q, c), L1(lam1, n=n)]
    if np.any(np.asarray(lam2) > 0):
        terms.append(L2Squared(lam2, n=n))
    p = Problem(terms)
    return p.subject_to(LinearInequality(A, k)) if A is not None else p
