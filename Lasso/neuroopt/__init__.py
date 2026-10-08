"""
neuroopt: LASSO (and related convex problems) solved on spiking neural networks.

    gram (quadratic)   -> recurrent weights I - gram
    linear             -> input current -linear
    l1 weight          -> dead zone
    l2 weight          -> firing threshold 2*l2 + 1
    linear constraints -> constraint neuron layer
    x >= 0             -> rectified neurons
"""

from .api import describe, elastic_net, lasso, network_for, nnls, quadratic_program, solve, sparse_qp
from .benchmark import ENERGY_PJ, benchmark, print_table
from .compile import Network, compile_network, rescale_to_unit_diagonal
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
    Term,
)
from .solvers import Result, solve_analog, solve_ista, solve_reference, solve_spiking
from .tuning import find_lambda, lambda_max

__version__ = "2.0.0"
