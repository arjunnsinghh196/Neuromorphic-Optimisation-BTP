"""
Problem definition.

Everything reduces to one canonical form:

    min  1/2 x'Gx + h'x + l1_weight'|x| + l2_weight'x^2
    s.t. constraint_matrix x <= constraint_rhs,  x >= 0 (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Term:
    size: int | None = None

    def __add__(self, other):
        return Problem([self]) + other


class Quadratic(Term):
    """1/2 x'Gx + h'x with G symmetric positive semi-definite."""

    def __init__(self, gram, linear=None):
        gram = np.asarray(gram, dtype=float)
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise ValueError("gram must be square")
        if not np.allclose(gram, gram.T, atol=1e-10):
            gram = (gram + gram.T) / 2

        self.gram = gram
        self.size = len(gram)
        self.linear = np.zeros(self.size) if linear is None else np.asarray(linear, dtype=float).ravel()
        if self.linear.shape != (self.size,):
            raise ValueError(f"linear must have length {self.size}")

        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues.min() < -1e-8 * max(1.0, abs(eigenvalues).max()):
            raise ValueError(f"gram is not PSD (min eigenvalue {eigenvalues.min():.3e})")

    def value(self, x):
        return 0.5 * x @ self.gram @ x + self.linear @ x


class LeastSquares(Quadratic):
    """1/2 ||signal - dictionary @ x||^2"""

    def __init__(self, dictionary, signal):
        dictionary = np.asarray(dictionary, dtype=float)
        signal = np.asarray(signal, dtype=float).ravel()
        if dictionary.shape[0] != signal.shape[0]:
            raise ValueError(f"dictionary has {dictionary.shape[0]} rows, signal has {len(signal)}")

        self.dictionary = dictionary
        self.signal = signal
        super().__init__(dictionary.T @ dictionary, -dictionary.T @ signal)

    def value(self, x):
        return 0.5 * np.sum((self.signal - self.dictionary @ x) ** 2)


class L1(Term):
    """weight * ||x||_1"""

    def __init__(self, weight, size=None):
        self.weight = np.asarray(weight, dtype=float)
        self.size = size or (self.weight.size if self.weight.ndim else None)

    def value(self, x):
        return float(np.sum(self.weight * np.abs(x)))


class L2Squared(Term):
    """weight * ||x||_2^2"""

    def __init__(self, weight, size=None):
        self.weight = np.asarray(weight, dtype=float)
        self.size = size or (self.weight.size if self.weight.ndim else None)

    def value(self, x):
        return float(np.sum(self.weight * x**2))


class Constraint:
    size: int | None = None


class LinearInequality(Constraint):
    """matrix @ x <= rhs"""

    def __init__(self, matrix, rhs):
        self.matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        self.rhs = np.asarray(rhs, dtype=float).ravel()
        if self.matrix.shape[0] != len(self.rhs):
            raise ValueError(f"matrix has {self.matrix.shape[0]} rows, rhs has {len(self.rhs)}")
        self.size = self.matrix.shape[1]


class NonNegative(Constraint):
    """x >= 0"""

    def __init__(self, size=None):
        self.size = size


class Box(Constraint):
    """low <= x <= high"""

    def __init__(self, low=None, high=None, size=None):
        self.low, self.high, self.size = low, high, size

    def as_inequality(self, size):
        rows, rhs = [], []
        if self.high is not None:
            rows.append(np.eye(size))
            rhs.append(np.broadcast_to(np.asarray(self.high, dtype=float), size))
        if self.low is not None:
            rows.append(-np.eye(size))
            rhs.append(-np.broadcast_to(np.asarray(self.low, dtype=float), size))
        if not rows:
            raise ValueError("Box needs low or high")
        return LinearInequality(np.vstack(rows), np.concatenate(rhs))


@dataclass
class CanonicalProblem:
    gram: np.ndarray
    linear: np.ndarray
    l1_weight: np.ndarray
    l2_weight: np.ndarray
    constraint_matrix: np.ndarray | None = None
    constraint_rhs: np.ndarray | None = None
    nonneg: bool = False
    source: Problem | None = field(default=None, repr=False)

    @property
    def size(self):
        return len(self.gram)

    @property
    def n_constraints(self):
        return 0 if self.constraint_matrix is None else len(self.constraint_matrix)

    @property
    def constrained(self):
        return self.constraint_matrix is not None

    def smooth_gradient(self, x):
        return self.gram @ x + self.linear + 2 * self.l2_weight * x

    def objective(self, x):
        if self.source is not None:
            return self.source.objective(x)
        return (0.5 * x @ self.gram @ x + self.linear @ x
                + np.sum(self.l1_weight * np.abs(x)) + np.sum(self.l2_weight * x**2))

    def violation(self, x):
        worst = 0.0
        if self.constraint_matrix is not None:
            worst = max(worst, float(np.max(self.constraint_matrix @ x - self.constraint_rhs, initial=0.0)))
        if self.nonneg:
            worst = max(worst, float(np.max(-x, initial=0.0)))
        return worst

    def kkt_residual(self, x, multipliers=None):
        gradient = self.smooth_gradient(x)
        if multipliers is not None and self.constraint_matrix is not None:
            gradient = gradient + self.constraint_matrix.T @ multipliers

        active = np.abs(x) > 1e-12
        residual = np.empty_like(gradient)
        residual[active] = gradient[active] + self.l1_weight[active] * np.sign(x[active])
        residual[~active] = np.maximum(np.abs(gradient[~active]) - self.l1_weight[~active], 0.0)
        return float(np.linalg.norm(residual))


class Problem:
    def __init__(self, terms, constraints=()):
        self.terms = list(terms)
        self.constraints = list(constraints)

    def __add__(self, other):
        if isinstance(other, Problem):
            return Problem(self.terms + other.terms, self.constraints + other.constraints)
        if isinstance(other, Term):
            return Problem(self.terms + [other], self.constraints)
        if isinstance(other, Constraint):
            return Problem(self.terms, self.constraints + [other])
        return NotImplemented

    def subject_to(self, *constraints):
        return Problem(self.terms, self.constraints + list(constraints))

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        return float(sum(term.value(x) for term in self.terms))

    def canonicalize(self):
        size = next((obj.size for obj in self.terms + self.constraints if obj.size), None)
        if size is None:
            raise ValueError("can't infer problem size; pass size= to a term")

        gram = np.zeros((size, size))
        linear = np.zeros(size)
        l1_weight = np.zeros(size)
        l2_weight = np.zeros(size)

        for term in self.terms:
            if isinstance(term, Quadratic):
                gram += term.gram
                linear += term.linear
            elif isinstance(term, L1):
                l1_weight += np.broadcast_to(term.weight, size)
            elif isinstance(term, L2Squared):
                l2_weight += np.broadcast_to(term.weight, size)
            else:
                raise TypeError(f"unsupported term {type(term).__name__}")

        if (l1_weight < 0).any() or (l2_weight < 0).any():
            raise ValueError("regularisation weights must be >= 0")

        rows, rhs, nonneg = [], [], False
        for constraint in self.constraints:
            if isinstance(constraint, Box):
                constraint = constraint.as_inequality(size)
            if isinstance(constraint, LinearInequality):
                if constraint.matrix.shape[1] != size:
                    raise ValueError(f"constraint has {constraint.matrix.shape[1]} columns, problem has {size}")
                rows.append(constraint.matrix)
                rhs.append(constraint.rhs)
            elif isinstance(constraint, NonNegative):
                nonneg = True
            else:
                raise TypeError(f"unsupported constraint {type(constraint).__name__}")

        return CanonicalProblem(
            gram, linear, l1_weight, l2_weight,
            np.vstack(rows) if rows else None,
            np.concatenate(rhs) if rhs else None,
            nonneg, source=self,
        )
