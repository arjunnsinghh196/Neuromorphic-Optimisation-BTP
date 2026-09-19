"""
Compile a :class:`CanonicalProblem` into a network of neurons.

The mapping implemented here is the one derived in the project notes, with two
corrections that matter for correctness:

1. **No self-synapse.**  The recurrent weight matrix is ``W = I - Q``, whose
   diagonal vanishes *only if* ``diag(Q) = 1``.  A neuron's own inhibition is its
   reset, not a synapse.  If the diagonal is not unity the compiler rescales the
   problem (see :func:`normalize`) rather than silently solving a different
   problem.

2. **Elastic-net gain.**  The l2 weight enters as the firing threshold
   ``nu_f = 2*lam2 + 1``, the l1 weight as the dead zone ``lam1``.  The synaptic
   weights are untouched by either.

Resulting dynamics (continuous time, one layer per role):

    primal:  u_dot = -u + W a - c - A^T v ,   a = shrink(u, lam1) / nu_f
    dual:    w_dot = beta (A a - k) ,         v = [w]_+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .objectives import CanonicalProblem

__all__ = ["NetworkSpec", "compile_network", "normalize"]


@dataclass
class Scaling:
    """``x_original = scale * x_network``."""

    scale: np.ndarray

    def to_network(self, x):
        return x / self.scale

    def to_original(self, x):
        return self.scale * x


@dataclass
class NetworkSpec:
    """Everything a solver backend needs: weights, biases, neuron parameters."""

    # primal (gradient-descent) layer
    W: np.ndarray            # recurrent weights, I - Q, zero diagonal
    bias: np.ndarray         # constant input current, -c
    lam1: np.ndarray         # dead zone / firing threshold offset
    nu_f: np.ndarray         # firing threshold, 2*lam2 + 1
    nonneg: bool             # rectified (one-sided) neurons

    # dual (constraint) layer
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

    # ---- neuron nonlinearity ---------------------------------------------- #
    def activation(self, u):
        """Thresholded readout ``a = shrink(u, lam1) / nu_f``."""
        if self.nonneg:
            return np.maximum(u - self.lam1, 0.0) / self.nu_f
        return np.sign(u) * np.maximum(np.abs(u) - self.lam1, 0.0) / self.nu_f

    def drive(self, u):
        """Soft-thresholded soma current that charges the membrane.

        This is the signed generalisation of the ``v_dot = mu - lambda`` rule in
        the notes: integrating ``mu - lambda`` only produces the right dead zone
        for non-negative currents.  Shrinking the current instead gives the exact
        soft-threshold f-I curve for both signs.
        """
        if self.nonneg:
            return np.maximum(u - self.lam1, 0.0)
        return np.sign(u) * np.maximum(np.abs(u) - self.lam1, 0.0)

    # ---- diagnostics ------------------------------------------------------- #
    def spectrum(self):
        ev = np.linalg.eigvalsh(self.Q + 2.0 * np.diag((self.nu_f - 1.0) / 2.0))
        lo, hi = float(ev.min()), float(ev.max())
        return {
            "lambda_min": lo,
            "lambda_max": hi,
            "condition_number": (hi / lo) if lo > 1e-14 else np.inf,
            "max_euler_step": 2.0 / hi if hi > 0 else np.inf,
        }


def normalize(prob: CanonicalProblem, atol: float = 1e-9):
    """Rescale variables so that ``diag(Q) = 1``.

    With ``x = S x_tilde``, ``S = diag(diag(Q)^{-1/2})``:

        Q~ = S Q S     (unit diagonal)      c~ = S c
        lam1~ = S lam1                      lam2~ = S^2 lam2
        A~ = A S                            k~ = k

    The l1 weight becomes per-coordinate even if it started scalar, which is why
    the whole library carries vector ``lam1``/``lam2``.  Without this step the
    ``-I`` in ``(Q - I)`` fails to cancel self-inhibition and each neuron ends up
    with an effective threshold ``lam1_i / Q_ii``.
    """
    d = np.diag(prob.Q).copy()
    if np.allclose(d, 1.0, atol=atol):
        return prob, Scaling(np.ones(prob.n)), False

    s = np.ones(prob.n)
    pos = d > 0
    s[pos] = 1.0 / np.sqrt(d[pos])  # scale factor applied to the *network* var
    S = np.diag(s)

    new = CanonicalProblem(
        Q=S @ prob.Q @ S,
        c=s * prob.c,
        lam1=s * prob.lam1,
        lam2=(s**2) * prob.lam2,
        A=None if prob.A is None else prob.A @ S,
        k=prob.k,
        nonneg=prob.nonneg,
        source=None,
    )
    np.fill_diagonal(new.Q, np.where(pos, 1.0, np.diag(new.Q)))
    return new, Scaling(s), True


def compile_network(prob: CanonicalProblem, auto_normalize: bool = True) -> NetworkSpec:
    """Map a canonical problem onto neurons, weights and thresholds."""
    scaling = Scaling(np.ones(prob.n))
    normalized = False
    if auto_normalize:
        prob_n, scaling, normalized = normalize(prob)
    else:
        prob_n = prob
        d = np.diag(prob.Q)
        if not np.allclose(d, 1.0, atol=1e-6):
            raise ValueError(
                "diag(Q) != 1 and auto_normalize=False; the -I self-inhibition "
                "cancellation is invalid.  Normalise the dictionary atoms or let "
                "the compiler rescale."
            )

    W = np.eye(prob_n.n) - prob_n.Q
    np.fill_diagonal(W, 0.0)  # a neuron never synapses onto itself

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
