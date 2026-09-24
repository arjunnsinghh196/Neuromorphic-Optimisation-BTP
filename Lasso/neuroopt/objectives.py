"""
How a problem is written down.

You build a problem as a sum of terms plus a list of constraints. Everything
then gets reduced to one canonical form that all the solvers understand:

    minimize   1/2 x'Qx + c'x + sum_i lam1_i |x_i| + sum_i lam2_i x_i^2
    s.t.       A x <= k
               x >= 0          (optional)

That's the widest class where the objective-to-neuron mapping is exact, so
anything outside it is rejected at canonicalize() time rather than silently
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "Term", "Quadratic", "LeastSquares", "L1", "L2Squared",
    "Constraint", "LinearInequality", "NonNegative", "Box",
    "Problem", "CanonicalProblem",
]


# ---------------------------------------------------------------------------
# objective terms
# ---------------------------------------------------------------------------
class Term:
    """Base class for objective terms. Terms add together into a Problem."""

    n: int

    def __add__(self, other: "Term") -> "Problem":
        return Problem([self]) + other

    def __radd__(self, other):
        # lets sum([...]) work, since sum starts from 0
        if other == 0:
            return Problem([self])
        return NotImplemented


class Quadratic(Term):
    """1/2 x'Qx + c'x, with Q symmetric PSD."""

    smooth = True

    def __init__(self, Q: np.ndarray, c: Optional[np.ndarray] = None):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square matrix")

        # Only the symmetric part contributes to x'Qx, so just symmetrise.
        if not np.allclose(Q, Q.T, atol=1e-10):
            Q = 0.5 * (Q + Q.T)

        self.Q = Q
        self.n = Q.shape[0]
        self.c = np.zeros(self.n) if c is None else np.asarray(c, dtype=float).ravel()
        if self.c.shape != (self.n,):
            raise ValueError(f"c should have length {self.n}, got {self.c.shape}")

        # Non-convex Q means the gradient flow can run away. Catch it early.
        w = np.linalg.eigvalsh(self.Q)
        if w.min() < -1e-8 * max(1.0, abs(w).max()):
            raise ValueError(
                f"Q is not positive semi-definite (smallest eigenvalue {w.min():.3e}). "
                "The problem is non-convex and none of the solvers here will work."
            )

    def value(self, x):
        return 0.5 * x @ self.Q @ x + self.c @ x

    def grad(self, x):
        return self.Q @ x + self.c


class LeastSquares(Quadratic):
    """1/2 ||s - Phi a||^2, which is just Q = Phi'Phi and c = -Phi's."""

    def __init__(self, Phi: np.ndarray, s: np.ndarray, normalize_atoms: bool = False):
        Phi = np.asarray(Phi, dtype=float)
        s = np.asarray(s, dtype=float).ravel()
        if Phi.shape[0] != s.shape[0]:
            raise ValueError(
                f"Phi has {Phi.shape[0]} rows but s has {s.shape[0]} entries"
            )

        self.atom_norms = np.linalg.norm(Phi, axis=0)
        if normalize_atoms:
            if np.any(self.atom_norms == 0):
                raise ValueError("dictionary has a zero column, can't normalise it")
            Phi = Phi / self.atom_norms

        self.Phi = Phi
        self.s = s
        super().__init__(Phi.T @ Phi, -(Phi.T @ s))

    def value(self, x):
        # Evaluate in the original form; it's better conditioned than x'Qx.
        return 0.5 * np.sum((self.s - self.Phi @ x) ** 2)


class L1(Term):
    """lam * ||x||_1. lam can be a scalar or one value per coordinate."""

    smooth = False

    def __init__(self, lam, n: Optional[int] = None):
        self.lam = np.asarray(lam, dtype=float)
        self.n = n if n is not None else (self.lam.size if self.lam.ndim else None)

    def value(self, x):
        return float(np.sum(np.broadcast_to(self.lam, x.shape) * np.abs(x)))


class L2Squared(Term):
    """lam * ||x||_2^2 - note the square.

    The unsquared norm has a completely different prox (it zeroes the whole
    vector at once) and needs a different neuron, so it isn't supported. The
    name is deliberately explicit so nobody reaches for it by accident.
    """

    smooth = True

    def __init__(self, lam, n: Optional[int] = None):
        self.lam = np.asarray(lam, dtype=float)
        self.n = n if n is not None else (self.lam.size if self.lam.ndim else None)

    def value(self, x):
        return float(np.sum(np.broadcast_to(self.lam, x.shape) * x ** 2))

    def grad(self, x):
        return 2.0 * np.broadcast_to(self.lam, x.shape) * x


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------
class Constraint:
    pass


class LinearInequality(Constraint):
    """A x <= k."""

    def __init__(self, A: np.ndarray, k: np.ndarray):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.k = np.asarray(k, dtype=float).ravel()
        if self.A.shape[0] != self.k.shape[0]:
            raise ValueError(
                f"A has {self.A.shape[0]} rows but k has {self.k.shape[0]} entries"
            )
        self.n = self.A.shape[1]


class NonNegative(Constraint):
    """x >= 0. Baked into the neuron (rectifier), not sent to the dual layer."""

    def __init__(self, n: Optional[int] = None):
        self.n = n


class Box(Constraint):
    """lo <= x <= hi. Turned into plain linear inequalities when canonicalised."""

    def __init__(self, lo=None, hi=None, n: Optional[int] = None):
        self.lo, self.hi, self.n = lo, hi, n

    def expand(self, n: int) -> LinearInequality:
        rows, rhs = [], []
        eye = np.eye(n)
        if self.hi is not None:
            rows.append(eye)
            rhs.append(np.broadcast_to(np.asarray(self.hi, dtype=float), (n,)))
        if self.lo is not None:
            rows.append(-eye)
            rhs.append(-np.broadcast_to(np.asarray(self.lo, dtype=float), (n,)))
        if not rows:
            raise ValueError("Box needs at least one of lo or hi")
        return LinearInequality(np.vstack(rows), np.concatenate(rhs))


# ---------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------
@dataclass
class CanonicalProblem:
    """The one form every solver accepts. See the module docstring."""

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

    # ---- evaluation --------------------------------------------------------
    def smooth_grad(self, x):
        """Gradient of the smooth part (everything except the l1 term)."""
        return self.Q @ x + self.c + 2.0 * self.lam2 * x

    def objective(self, x):
        # Prefer the original terms if we have them; LeastSquares evaluates
        # more accurately in its own form than via Q.
        if self.source is not None:
            return self.source.objective(x)
        return (
            0.5 * x @ self.Q @ x
            + self.c @ x
            + np.sum(self.lam1 * np.abs(x))
            + np.sum(self.lam2 * x ** 2)
        )

    def violation(self, x):
        """Largest constraint violation; 0 if x is feasible."""
        v = 0.0
        if self.A is not None:
            v = max(v, float(np.max(self.A @ x - self.k, initial=0.0)))
        if self.nonneg:
            v = max(v, float(np.max(-x, initial=0.0)))
        return v

    def kkt_residual(self, x, y=None):
        """How far x is from satisfying the optimality conditions.

        For coordinates that are zero, the subgradient of |x| is the whole
        interval [-lam1, lam1], so the residual is the distance of -g to that
        interval. For non-zero coordinates it's simply |g + lam1 * sign(x)|.
        Pass y (dual multipliers) if the problem has constraints.
        """
        g = self.smooth_grad(x)
        if y is not None and self.A is not None:
            g = g + self.A.T @ y

        active = np.abs(x) > 1e-12
        r = np.empty_like(g)
        r[active] = g[active] + self.lam1[active] * np.sign(x[active])
        r[~active] = np.maximum(np.abs(g[~active]) - self.lam1[~active], 0.0)
        return float(np.linalg.norm(r))


class Problem:
    """A list of terms and a list of constraints, nothing more."""

    def __init__(self, terms: Sequence[Term], constraints: Sequence[Constraint] = ()):
        self.terms: List[Term] = list(terms)
        self.constraints: List[Constraint] = list(constraints)

    # ---- composition -------------------------------------------------------
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

    # ---- evaluation --------------------------------------------------------
    def objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(sum(t.value(x) for t in self.terms))

    # ---- reduction ---------------------------------------------------------
    def _infer_dimension(self) -> int:
        for t in self.terms:
            if getattr(t, "n", None):
                return t.n
        for con in self.constraints:
            if getattr(con, "n", None):
                return con.n
        raise ValueError(
            "couldn't work out the problem dimension; pass n= to one of the terms"
        )

    def canonicalize(self) -> CanonicalProblem:
        n = self._infer_dimension()

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
                raise TypeError(f"don't know how to handle term {type(t).__name__}")

        if np.any(lam1 < 0) or np.any(lam2 < 0):
            raise ValueError("regularisation weights have to be >= 0")

        rows, rhs, nonneg = [], [], False
        for con in self.constraints:
            if isinstance(con, Box):
                con = con.expand(n)
            if isinstance(con, LinearInequality):
                if con.A.shape[1] != n:
                    raise ValueError(
                        f"constraint has {con.A.shape[1]} columns, problem has {n} variables"
                    )
                rows.append(con.A)
                rhs.append(con.k)
            elif isinstance(con, NonNegative):
                nonneg = True
            else:
                raise TypeError(f"don't know how to handle constraint {type(con).__name__}")

        A = np.vstack(rows) if rows else None
        k = np.concatenate(rhs) if rhs else None
        return CanonicalProblem(Q, c, lam1, lam2, A, k, nonneg, source=self)