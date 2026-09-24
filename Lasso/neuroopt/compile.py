"""
Turn a CanonicalProblem into a network: weights, biases, thresholds.

Two details here are easy to get wrong and both matter:

1. No self-synapse. The recurrent weights are W = I - Q, and that diagonal is
   only zero when diag(Q) = 1. A neuron's self-inhibition is its reset, not a
   synapse, so if the diagonal isn't 1 we rescale the variables first
   (see normalize) instead of quietly solving a slightly different problem.

2. The l2 term doesn't touch the weights. It shows up as the firing threshold
   nu_f = 2*lam2 + 1. The l1 term is the dead zone lam1. That's it.

The dynamics every backend implements, in continuous time:

    primal:  u' = -u + W a - c - A'v        a = shrink(u, lam1) / nu_f
    dual:    w' = beta (A a - k)            v = max(w, 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .objectives import CanonicalProblem

__all__ = ["NetworkSpec", "compile_network", "normalize"]


@dataclass
class Scaling:
    """Maps between the user's variables and the network's. x_user = scale * x_net."""

    scale: np.ndarray

    def to_network(self, x):
        return x / self.scale

    def to_original(self, x):
        return self.scale * x


@dataclass
class NetworkSpec:
    """Everything a solver needs to run the network."""

    # primal layer (one neuron per variable)
    W: np.ndarray        # recurrent weights, I - Q with the diagonal zeroed
    bias: np.ndarray     # constant input current, -c
    lam1: np.ndarray     # dead zone
    nu_f: np.ndarray     # firing threshold, 2*lam2 + 1
    nonneg: bool         # rectified neurons?

    # dual layer (one neuron per constraint), or None
    A: Optional[np.ndarray] = None
    k: Optional[np.ndarray] = None

    # bookkeeping
    Q: np.ndarray = field(default=None, repr=False)
    scaling: Scaling = field(default=None, repr=False)
    problem: CanonicalProblem = field(default=None, repr=False)
    normalized: bool = False

    @property
    def n(self):
        return self.W.shape[0]

    @property
    def m(self):
        return 0 if self.A is None else self.A.shape[0]

    # ---- neuron nonlinearity -----------------------------------------------
    def drive(self, u):
        """Soft-thresholded soma current: shrink(u, lam1).

        Shrinking the current (rather than integrating u - lam1 and clipping)
        gives the right dead zone for both signs, not just positive currents.
        """
        if self.nonneg:
            return np.maximum(u - self.lam1, 0.0)
        return np.sign(u) * np.maximum(np.abs(u) - self.lam1, 0.0)

    def activation(self, u):
        """Readout a = shrink(u, lam1) / nu_f."""
        return self.drive(u) / self.nu_f

    # ---- spectrum, used to pick step sizes ---------------------------------
    def spectrum(self):
        # Effective Gram matrix including the l2 shift: Q + 2*lam2*I
        ev = np.linalg.eigvalsh(self.Q + np.diag(self.nu_f - 1.0))
        lo, hi = float(ev.min()), float(ev.max())
        return {
            "lambda_min": lo,
            "lambda_max": hi,
            "condition_number": (hi / lo) if lo > 1e-14 else np.inf,
            "max_euler_step": 2.0 / hi if hi > 0 else np.inf,
        }


def normalize(prob: CanonicalProblem, atol: float = 1e-9):
    """Rescale variables so diag(Q) = 1.

    With x = S x~ and S = diag(Q)^(-1/2):

        Q~ = S Q S      c~ = S c      lam1~ = S lam1      lam2~ = S^2 lam2
        A~ = A S        k~ = k

    Note lam1 becomes per-coordinate even if it started out scalar, which is
    why lam1/lam2 are vectors throughout the library.

    Returns (new_problem, scaling, did_anything).
    """
    d = np.diag(prob.Q).copy()
    if np.allclose(d, 1.0, atol=atol):
        return prob, Scaling(np.ones(prob.n)), False

    s = np.ones(prob.n)
    pos = d > 0
    s[pos] = 1.0 / np.sqrt(d[pos])
    S = np.diag(s)

    new = CanonicalProblem(
        Q=S @ prob.Q @ S,
        c=s * prob.c,
        lam1=s * prob.lam1,
        lam2=(s ** 2) * prob.lam2,
        A=None if prob.A is None else prob.A @ S,
        k=prob.k,
        nonneg=prob.nonneg,
        source=None,
    )
    # Pin the diagonal to exactly 1 where we could; floating point otherwise
    # leaves it at 0.99999999.
    np.fill_diagonal(new.Q, np.where(pos, 1.0, np.diag(new.Q)))
    return new, Scaling(s), True


def compile_network(prob: CanonicalProblem, auto_normalize: bool = True) -> NetworkSpec:
    """Map a canonical problem onto neurons."""
    if auto_normalize:
        prob_n, scaling, normalized = normalize(prob)
    else:
        if not np.allclose(np.diag(prob.Q), 1.0, atol=1e-6):
            raise ValueError(
                "diag(Q) != 1 with auto_normalize=False. The self-inhibition "
                "cancellation in W = I - Q won't hold. Either normalise your "
                "dictionary columns or let the compiler rescale."
            )
        prob_n, scaling, normalized = prob, Scaling(np.ones(prob.n)), False

    W = np.eye(prob_n.n) - prob_n.Q
    np.fill_diagonal(W, 0.0)   # never a synapse onto itself

    return NetworkSpec(
        W=W,
        bias=-prob_n.c,
        lam1=prob_n.lam1.copy(),
        nu_f=2.0 * prob_n.lam2 + 1.0,
        nonneg=prob_n.nonneg,
        A=prob_n.A,
        k=prob_n.k,
        Q=prob_n.Q,
        scaling=scaling,
        problem=prob,
        normalized=normalized,
    )