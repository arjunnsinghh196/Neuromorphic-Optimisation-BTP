"""
Public API.

    problem = lasso(dictionary, signal, lam=0.1)
    print(describe(problem))
    result = solve(problem, backend="spiking")
"""

from __future__ import annotations

import numpy as np

from .compile import compile_network
from .objectives import (
    L1,
    Box,
    CanonicalProblem,
    L2Squared,
    LeastSquares,
    LinearInequality,
    NonNegative,
    Problem,
    Quadratic,
)
from .solvers import solve_analog, solve_ista, solve_reference, solve_spiking

BACKENDS = {
    "spiking": solve_spiking,
    "analog": solve_analog,
    "ista": solve_ista,
    "fista": lambda problem, **kw: solve_ista(problem, accelerate=True, **kw),
    "reference": solve_reference,
}


def _canonical(problem):
    return problem if isinstance(problem, CanonicalProblem) else problem.canonicalize()


def solve(problem, backend="spiking", **kwargs):
    problem = _canonical(problem)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; options: {sorted(BACKENDS)}")
    if backend in ("ista", "fista") and problem.constraint_matrix is not None:
        raise ValueError("ista/fista don't support constraints; use spiking or analog")
    return BACKENDS[backend](problem, **kwargs)


def network_for(problem, **kwargs):
    return compile_network(_canonical(problem), **kwargs)


def describe(problem):
    problem = _canonical(problem)
    network = compile_network(problem)
    spectrum = network.spectrum()
    has_l1 = (problem.l1_weight > 0).any()
    has_l2 = (problem.l2_weight > 0).any()

    if problem.constrained:
        neuron = "two-layer: IF primal + constraint neurons"
        fit = "good - constraint neurons only spike while active"
    elif has_l1 and has_l2:
        neuron = "IF with dead zone and reduced gain"
        fit = "good"
    elif has_l1:
        neuron = "integrate-and-fire with dead zone"
        fit = "excellent - sparse solution, sparse spikes"
    elif has_l2:
        neuron = "linear integrator with extra leak"
        fit = "poor - dense solution, every neuron fires"
    else:
        neuron = "linear leaky integrator"
        fit = "poor - no sparsity"

    def show(v):
        return f"{v[0]:.4g}" if np.allclose(v, v[0]) else f"[{v.min():.4g}, {v.max():.4g}]"

    lines = [
        "objective -> neural architecture",
        "-" * 52,
        f"decision variables      : {problem.size}",
        f"linear constraints      : {problem.n_constraints}",
        f"neuron model            : {neuron}",
        f"recurrent weights       : I - gram  ({network.n_synapses} synapses)",
        f"input current           : -linear",
        f"dead zone (l1 weight)   : {show(problem.l1_weight)}",
        f"threshold (2*l2 + 1)    : {show(network.threshold)}",
        f"rectified neurons       : {problem.nonneg}",
        f"variable rescaling      : {'applied' if network.rescaled else 'not needed'}",
        "",
        "convergence",
        "-" * 52,
        f"lambda_min              : {spectrum['lambda_min']:.4g}",
        f"lambda_max              : {spectrum['lambda_max']:.4g}",
        f"condition number        : {spectrum['condition_number']:.4g}",
        f"max stable Euler step   : {spectrum['max_euler_step']:.4g}",
        f"unit-step Euler (=ISTA) : {'stable' if spectrum['lambda_max'] < 2 else 'UNSTABLE'}",
        f"hardware fit            : {fit}",
    ]
    if spectrum["lambda_min"] <= 1e-10:
        lines += ["", "note: gram is rank deficient (normal when atoms > samples);",
                  "      convergence rate is set by lambda_min on the active set."]
    elif spectrum["condition_number"] > 1e3:
        lines += ["", "note: gram is ill-conditioned; settling time ~ 1/lambda_min."]
    return "\n".join(lines)


def _add_constraints(problem, constraints, size, nonneg):
    constraints = list(constraints) + ([NonNegative(size)] if nonneg else [])
    return problem.subject_to(*constraints) if constraints else problem


def lasso(dictionary, signal, lam, nonneg=False, constraints=()):
    """1/2 ||signal - dictionary @ x||^2 + lam ||x||_1"""
    size = np.asarray(dictionary).shape[1]
    problem = Problem([LeastSquares(dictionary, signal), L1(lam, size=size)])
    return _add_constraints(problem, constraints, size, nonneg)


def elastic_net(dictionary, signal, l1, l2, nonneg=False, constraints=()):
    """1/2 ||signal - dictionary @ x||^2 + l1 ||x||_1 + l2 ||x||_2^2"""
    size = np.asarray(dictionary).shape[1]
    problem = Problem([LeastSquares(dictionary, signal), L1(l1, size=size), L2Squared(l2, size=size)])
    return _add_constraints(problem, constraints, size, nonneg)


def nnls(dictionary, signal, lam=0.0):
    """Non-negative LASSO."""
    return lasso(dictionary, signal, lam, nonneg=True)


def quadratic_program(gram, linear, matrix=None, rhs=None, low=None, high=None, nonneg=False):
    """1/2 x'Gx + h'x  s.t.  matrix @ x <= rhs, low <= x <= high"""
    size = len(gram)
    constraints = []
    if matrix is not None:
        constraints.append(LinearInequality(matrix, rhs))
    if low is not None or high is not None:
        constraints.append(Box(low, high, size=size))
    if nonneg:
        constraints.append(NonNegative(size))
    return Problem([Quadratic(gram, linear)], constraints)


def sparse_qp(gram, linear, l1, matrix=None, rhs=None, l2=0.0):
    """1/2 x'Gx + h'x + l1 ||x||_1 + l2 ||x||_2^2  s.t.  matrix @ x <= rhs"""
    size = len(gram)
    terms = [Quadratic(gram, linear), L1(l1, size=size)]
    if np.any(np.asarray(l2) > 0):
        terms.append(L2Squared(l2, size=size))
    constraints = [LinearInequality(matrix, rhs)] if matrix is not None else []
    return Problem(terms, constraints)
