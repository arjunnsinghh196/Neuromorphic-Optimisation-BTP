"""
neuroopt -- convex optimisation as spiking neural dynamics.

The library takes an optimisation objective and a set of constraints, and emits a
neuron model, a synaptic weight structure, and an expected convergence behaviour.
The mapping is exact, not metaphorical:

    Q (quadratic)   -> recurrent synaptic weights      W = I - Q
    c (linear)      -> constant bias current           -c
    lam1 (l1)       -> dead zone / firing threshold offset
    lam2 (l2^2)     -> firing threshold nu_f = 2*lam2 + 1  (gain 1/nu_f)
    Ax <= k         -> second neuron layer, feedback through A^T
    x >= 0          -> rectified (one-sided) neurons

Measured behaviour
------------------
The subtractive reset makes every neuron a first-order sigma-delta modulator, so
the spiking readout error scales as ``1/gamma`` (measured exponent -1.010), not
``1/sqrt(gamma)``.  The error is deterministic and non-monotone in gamma, and it
is caused entirely by lateral coupling: with ``W = 0`` the spiking solver matches
the continuous flow to machine precision at any spike budget.

Quick start
-----------
>>> import numpy as np, neuroopt as no
>>> rng = np.random.default_rng(0)
>>> Phi = rng.standard_normal((40, 80)); Phi /= np.linalg.norm(Phi, axis=0)
>>> a = np.zeros(80); a[[3, 17, 52]] = [2.0, -1.5, 0.8]
>>> s = Phi @ a
>>> res = no.solve(no.lasso(Phi, s, lam=0.05), backend="spiking")
>>> res.sparsity > 0.9
True
"""

from .api import (
    describe,
    elastic_net,
    lasso,
    network_for,
    nnls,
    quadratic_program,
    ridge,
    solve,
    sparse_qp,
)
from .compile import NetworkSpec, compile_network, normalize
from .diagnostics import compare, conditioning_report, spike_report
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
    Term,
)
from .solvers import Result, solve_analog, solve_ista, solve_reference, solve_spiking

__version__ = "1.0.0"

__all__ = [
    "solve", "describe", "network_for",
    "lasso", "elastic_net", "ridge", "nnls", "quadratic_program", "sparse_qp",
    "Problem", "CanonicalProblem", "Term", "Quadratic", "LeastSquares",
    "L1", "L2Squared", "LinearInequality", "NonNegative", "Box",
    "NetworkSpec", "compile_network", "normalize",
    "Result", "solve_spiking", "solve_analog", "solve_ista", "solve_reference",
    "compare", "conditioning_report", "spike_report",
]
