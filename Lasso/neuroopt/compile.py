"""
CanonicalProblem -> Network (weights, input current, dead zone, threshold).

Dynamics:
    membrane' = -membrane + weights @ activation + input_current - constraint_matrix' @ multipliers
    activation = shrink(membrane, dead_zone) / threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .objectives import CanonicalProblem


@dataclass
class Scaling:
    factor: np.ndarray  # x_user = factor * x_network

    def to_network(self, x):
        return x / self.factor

    def to_user(self, x):
        return self.factor * x


@dataclass
class Network:
    weights: np.ndarray          # I - gram, zero diagonal
    input_current: np.ndarray    # -linear
    dead_zone: np.ndarray        # l1 weight
    threshold: np.ndarray        # 2 * l2 weight + 1
    rectified: bool
    constraint_matrix: np.ndarray | None = None
    constraint_rhs: np.ndarray | None = None
    gram: np.ndarray = field(default=None, repr=False)
    scaling: Scaling = field(default=None, repr=False)
    problem: CanonicalProblem = field(default=None, repr=False)
    rescaled: bool = False

    @property
    def size(self):
        return len(self.weights)

    @property
    def n_constraints(self):
        return 0 if self.constraint_matrix is None else len(self.constraint_matrix)

    @property
    def n_synapses(self):
        return int(np.count_nonzero(self.weights))

    def shrink(self, membrane):
        if self.rectified:
            return np.maximum(membrane - self.dead_zone, 0.0)
        return np.sign(membrane) * np.maximum(np.abs(membrane) - self.dead_zone, 0.0)

    def activation(self, membrane):
        return self.shrink(membrane) / self.threshold

    def spectrum(self):
        eigenvalues = np.linalg.eigvalsh(self.gram + np.diag(self.threshold - 1))
        smallest, largest = float(eigenvalues.min()), float(eigenvalues.max())
        return {
            "lambda_min": smallest,
            "lambda_max": largest,
            "condition_number": largest / smallest if smallest > 1e-14 else np.inf,
            "max_euler_step": 2 / largest if largest > 0 else np.inf,
        }


def rescale_to_unit_diagonal(problem, atol=1e-9):
    diagonal = np.diag(problem.gram)
    if np.allclose(diagonal, 1.0, atol=atol):
        return problem, Scaling(np.ones(problem.size)), False

    positive = diagonal > 0
    factor = np.where(positive, 1 / np.sqrt(np.where(positive, diagonal, 1)), 1.0)
    S = np.diag(factor)
    gram = S @ problem.gram @ S
    np.fill_diagonal(gram, np.where(positive, 1.0, np.diag(gram)))

    scaled = CanonicalProblem(
        gram=gram,
        linear=factor * problem.linear,
        l1_weight=factor * problem.l1_weight,
        l2_weight=factor**2 * problem.l2_weight,
        constraint_matrix=None if problem.constraint_matrix is None else problem.constraint_matrix @ S,
        constraint_rhs=problem.constraint_rhs,
        nonneg=problem.nonneg,
    )
    return scaled, Scaling(factor), True


def compile_network(problem, auto_rescale=True):
    # weights = I - gram only has a zero diagonal when diag(gram) = 1.
    # A neuron's self-inhibition is its reset, not a synapse.
    if auto_rescale:
        scaled, scaling, rescaled = rescale_to_unit_diagonal(problem)
    elif np.allclose(np.diag(problem.gram), 1.0, atol=1e-6):
        scaled, scaling, rescaled = problem, Scaling(np.ones(problem.size)), False
    else:
        raise ValueError("diag(gram) != 1; normalise the dictionary or use auto_rescale=True")

    weights = np.eye(scaled.size) - scaled.gram
    np.fill_diagonal(weights, 0.0)

    return Network(
        weights=weights,
        input_current=-scaled.linear,
        dead_zone=scaled.l1_weight.copy(),
        threshold=2 * scaled.l2_weight + 1,
        rectified=scaled.nonneg,
        constraint_matrix=scaled.constraint_matrix,
        constraint_rhs=scaled.constraint_rhs,
        gram=scaled.gram,
        scaling=scaling,
        problem=problem,
        rescaled=rescaled,
    )
