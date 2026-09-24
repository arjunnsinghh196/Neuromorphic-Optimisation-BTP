"""
neuroopt - solve convex problems with spiking neurons.

You give it an objective and some constraints. It gives you back a neuron model,
a weight matrix, and a solver that runs that network. The mapping is literal:

    Q  (quadratic)      ->  recurrent weights        W = I - Q
    c  (linear)         ->  constant bias current    -c
    lam1 (l1 penalty)   ->  dead zone / threshold offset
    lam2 (squared l2)   ->  firing threshold nu_f = 2*lam2 + 1
    A x <= k            ->  a second layer of constraint neurons
    x >= 0              ->  one-sided (rectified) neurons

A quick example:

    >>> import numpy as np, neuroopt as no
    >>> rng = np.random.default_rng(0)
    >>> Phi = rng.standard_normal((40, 80)); Phi /= np.linalg.norm(Phi, axis=0)
    >>> a = np.zeros(80); a[[3, 17, 52]] = [2.0, -1.5, 0.8]
    >>> s = Phi @ a
    >>> res = no.solve(no.lasso(Phi, s, lam=0.05), backend="spiking")
    >>> res.sparsity > 0.9
    True

Worth knowing before you tune anything: the spiking readout error goes like
1/gamma (gamma = spikes per unit activation), not 1/sqrt(gamma), because the
subtractive reset turns each neuron into a sigma-delta modulator. The error
comes from lateral coupling only - with W = 0 the spiking run matches the
analog one to machine precision.
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

__version__ = "1.1.0"

__all__ = [
    # entry points
    "solve", "describe", "network_for",
    # ready-made problems
    "lasso", "elastic_net", "ridge", "nnls", "quadratic_program", "sparse_qp",
    # building blocks
    "Problem", "CanonicalProblem", "Term", "Quadratic", "LeastSquares",
    "L1", "L2Squared", "LinearInequality", "NonNegative", "Box",
    # compiler
    "NetworkSpec", "compile_network", "normalize",
    # solvers
    "Result", "solve_spiking", "solve_analog", "solve_ista", "solve_reference",
]