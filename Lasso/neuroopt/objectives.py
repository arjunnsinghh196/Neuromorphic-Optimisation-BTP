"""
Objective / constraint specification.

The user writes an optimisation problem as a sum of convex terms plus a list of
constraints.  ``Problem.canonicalize()`` reduces it to the single canonical form
that every solver in this library understands:

    minimize    1/2 x^T Q x + c^T x + sum_i lam1_i |x_i| + sum_i lam2_i x_i^2
    subject to  A x <= k
                x >= 0            (optional)

This is the widest class for which the objective -> neuron mapping is exact:

    Q       -> recurrent synaptic weights          (I - Q)
    c       -> constant bias current               (-c)
    lam1    -> firing threshold / dead zone        (per-neuron)
    lam2    -> firing threshold scale nu_f = 2*lam2 + 1  (per-neuron gain)
    A, k    -> second (constraint) neuron layer, feedback via A^T
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "Term",
    "Quadratic",
    "LeastSquares",
    "L1",
    "L2Squared",
    "Constraint",
    "LinearInequality",
    "NonNegative",
    "Box",
    "Problem",
    "CanonicalProblem",
]


# --------------------------------------------------------------------------- #
# objective terms
# --------------------------------------------------------------------------- #
class Term:
    """Base class for convex objective terms.  Terms compose with ``+``."""

    n: int  # problem dimension

    def __add__(self, other: "Term") -> "Problem":
        return Problem([self]) + other

    def __radd__(self, other):
        if other == 0:
            return Problem([self])
        return NotImplemented


class Quadratic(Term):
    """``1/2 x^T Q x + c^T x`` with Q symmetric positive semi-definite."""

    smooth = True

    def __init__(self, Q: np.ndarray, c: Optional[np.ndarray] = None):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square")
        if not np.allclose(Q, Q.T, atol=1e-10):
            Q = 0.5 * (Q + Q.T)  # symmetrise silently; only the sym part matters
        self.Q = Q
        self.n = Q.shape[0]
        self.c = np.zeros(self.n) if c is None else np.asarray(c, dtype=float).ravel()
        if self.c.shape != (self.n,):
            raise ValueError("c has wrong length")

        w = np.linalg.eigvalsh(self.Q)
        if w.min() < -1e-8 * max(1.0, abs(w).max()):
            raise ValueError(
                f"Q is not positive semi-definite (lambda_min = {w.min():.3e}); "
                "the problem is non-convex and the gradient-flow guarantees fail."
            )

    def value(self, x):
        return 0.5 * x @ self.Q @ x + self.c @ x

    def grad(self, x):
        return self.Q @ x + self.c


class LeastSquares(Quadratic):
    """``1/2 ||s - Phi a||^2``.  Expands to Q = Phi^T Phi, c = -Phi^T s."""

    def __init__(self, Phi: np.ndarray, s: np.ndarray, normalize_atoms: bool = False):
        Phi = np.asarray(Phi, dtype=float)
        s = np.asarray(s, dtype=float).ravel()
        if Phi.shape[0] != s.shape[0]:
            raise ValueError("Phi and s have incompatible shapes")
        self.atom_norms = np.linalg.norm(Phi, axis=0)
        if normalize_atoms:
            if np.any(self.atom_norms == 0):
                raise ValueError("dictionary contains a zero atom")
            Phi = Phi / self.atom_norms
        self.Phi = Phi
        self.s = s
        super().__init__(Phi.T @ Phi, -(Phi.T @ s))
        self._const = 0.5 * s @ s

    def value(self, x):
        return 0.5 * np.sum((self.s - self.Phi @ x) ** 2)


class L1(Term):
    """``lam * ||x||_1``.  ``lam`` may be scalar or per-coordinate."""

    smooth = False

    def __init__(self, lam, n: Optional[int] = None):
        self.lam = np.asarray(lam, dtype=float)
        self.n = n if n is not None else (self.lam.size if self.lam.ndim else None)

    def value(self, x):
        return float(np.sum(np.broadcast_to(self.lam, x.shape) * np.abs(x)))


class L2Squared(Term):
    """``lam * ||x||_2^2``  -- note the square.

    The unsquared ``||x||_2`` is *not* what you want here: its proximal operator
    is block soft-thresholding, which zeroes the whole vector at once and needs a
    different neuron.  This class is deliberately named to make that explicit.
    """

    smooth = True

    def __init__(self, lam, n: Optional[int] = None):
        self.lam = np.asarray(lam, dtype=float)
        self.n = n if n is not None else (self.lam.size if self.lam.ndim else None)

    def value(self, x):
        return float(np.sum(np.broadcast_to(self.lam, x.shape) * x**2))

    def grad(self, x):
        return 2.0 * np.broadcast_to(self.lam, x.shape) * x


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
class Constraint:
    pass


class LinearInequality(Constraint):
    """``A x <= k``."""

    def __init__(self, A: np.ndarray, k: np.ndarray):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.k = np.asarray(k, dtype=float).ravel()
        if self.A.shape[0] != self.k.shape[0]:
            raise ValueError("A and k have incompatible shapes")
        self.n = self.A.shape[1]


class NonNegative(Constraint):
    """``x >= 0``.  Handled by the neuron nonlinearity, not by the dual layer."""

    def __init__(self, n: Optional[int] = None):
        self.n = n


class Box(Constraint):
    """``lo <= x <= hi``; expanded into linear inequalities."""

    def __init__(self, lo=None, hi=None, n: Optional[int] = None):
        self.lo, self.hi, self.n = lo, hi, n

    def expand(self, n: int) -> LinearInequality:
        rows, rhs = [], []
        eye = np.eye(n)
        if self.hi is not None:
            hi = np.broadcast_to(np.asarray(self.hi, dtype=float), (n,))
            rows.append(eye)
            rhs.append(hi)
        if self.lo is not None:
            lo = np.broadcast_to(np.asarray(self.lo, dtype=float), (n,))
            rows.append(-eye)
            rhs.append(-lo)
        if not rows:
            raise ValueError("Box needs at least one of lo/hi")
        return LinearInequality(np.vstack(rows), np.concatenate(rhs))


# --------------------------------------------------------------------------- #
# canonical form
# --------------------------------------------------------------------------- #
@dataclass
class CanonicalProblem:
    """minimize 1/2 x'Qx + c'x + lam1'|x| + lam2'x^2  s.t. Ax <= k, (x >= 0)."""

    Q: np.ndarray
    c: np.ndarray
    lam1: np.ndarray
    lam2: np.ndarray
    A: Optional[np.ndarray] = None
    k: Optional[np.ndarray] = None
    nonneg: bool = False
    source: Optional["Problem"] = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def m(self) -> int:
        return 0 if self.A is None else self.A.shape[0]

    @property
    def constrained(self) -> bool:
        return self.A is not None

    # ---- evaluation -------------------------------------------------------- #
    def smooth_grad(self, x):
        return self.Q @ x + self.c + 2.0 * self.lam2 * x

    def objective(self, x):
        if self.source is not None:
            return self.source.objective(x)
        return (
            0.5 * x @ self.Q @ x
            + self.c @ x
            + np.sum(self.lam1 * np.abs(x))
            + np.sum(self.lam2 * x**2)
        )

    def violation(self, x):
        """Max constraint violation (0 when feasible)."""
        v = 0.0
        if self.A is not None:
            v = max(v, float(np.max(self.A @ x - self.k, initial=0.0)))
        if self.nonneg:
            v = max(v, float(np.max(-x, initial=0.0)))
        return v

    def kkt_residual(self, x, y=None):
        """Norm of the stationarity residual of the (sub)gradient conditions.

        For inactive coordinates the residual is the distance of ``-g_i`` to the
        interval ``[-lam1_i, lam1_i]``; for active ones it is ``|g_i + lam1_i
        sign(x_i)|``.  ``y`` are the dual multipliers, if any.
        """
        g = self.smooth_grad(x)
        if y is not None and self.A is not None:
            g = g + self.A.T @ y
        act = np.abs(x) > 1e-12
        r = np.empty_like(g)
        r[act] = g[act] + self.lam1[act] * np.sign(x[act])
        r[~act] = np.maximum(np.abs(g[~act]) - self.lam1[~act], 0.0)
        return float(np.linalg.norm(r))


class Problem:
    """A sum of :class:`Term` objects plus a list of :class:`Constraint`."""

    def __init__(self, terms: Sequence[Term], constraints: Sequence[Constraint] = ()):
        self.terms: List[Term] = list(terms)
        self.constraints: List[Constraint] = list(constraints)

    # ---- composition ------------------------------------------------------- #
    def __add__(self, other):
        if isinstance(other, Problem):
            return Problem(self.terms + other.terms, self.constraints + other.constraints)
        if isinstance(other, Term):
            return Problem(self.terms + [other], self.constraints)
        if isinstance(other, Constraint):
            return Problem(self.terms, self.constraints + [other])
        return NotImplemented

    def subject_to(self, *constraints: Constraint) -> "Problem":
        return Problem(self.terms, self.constraints + list(constraints))

    # ---- evaluation -------------------------------------------------------- #
    def objective(self, x):
        return float(sum(t.value(np.asarray(x, dtype=float)) for t in self.terms))

    # ---- reduction --------------------------------------------------------- #
    def canonicalize(self) -> CanonicalProblem:
        n = None
        for t in self.terms:
            if getattr(t, "n", None):
                n = t.n
                break
        if n is None:
            for con in self.constraints:
                if getattr(con, "n", None):
                    n = con.n
                    break
        if n is None:
            raise ValueError("cannot infer problem dimension; pass n= to a term")

        Q = np.zeros((n, n))
        c = np.zeros(n)
        lam1 = np.zeros(n)
        lam2 = np.zeros(n)

        for t in self.terms:
            if isinstance(t, Quadratic):
                Q += t.Q
                c += t.c
            elif isinstance(t, L1):
                lam1 += np.broadcast_to(t.lam, (n,))
            elif isinstance(t, L2Squared):
                lam2 += np.broadcast_to(t.lam, (n,))
            else:
                raise TypeError(f"unsupported objective term: {type(t).__name__}")

        if np.any(lam1 < 0) or np.any(lam2 < 0):
            raise ValueError("regularisation weights must be non-negative")

        rows, rhs, nonneg = [], [], False
        for con in self.constraints:
            if isinstance(con, Box):
                con = con.expand(n)
            if isinstance(con, LinearInequality):
                if con.A.shape[1] != n:
                    raise ValueError("constraint dimension mismatch")
                rows.append(con.A)
                rhs.append(con.k)
            elif isinstance(con, NonNegative):
                nonneg = True
            else:
                raise TypeError(f"unsupported constraint: {type(con).__name__}")

        A = np.vstack(rows) if rows else None
        k = np.concatenate(rhs) if rhs else None
        return CanonicalProblem(Q, c, lam1, lam2, A, k, nonneg, source=self)
