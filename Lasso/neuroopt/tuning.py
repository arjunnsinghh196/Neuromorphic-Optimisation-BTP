"""
Pick the l1 weight (lambda) for LASSO automatically.

Sweeps lambda from large to small, warm-starting each solve from the previous
one. Each solution's active set is refit by least squares (relaxed LASSO) so
the score reflects the chosen atoms, not the shrinkage, and then scored with
AIC or BIC:

    variance = RSS_refit / (n_samples - n_active)
    BIC = n_samples * log(variance) + n_active * log(n_samples)
    AIC = n_samples * log(variance) + 2 * n_active

Lower is better. Both reward a good fit and punish extra non-zeros. The
degrees-of-freedom correction stops the score from going to -inf when a
square or overcomplete dictionary fits the signal exactly.
"""

from __future__ import annotations

import numpy as np

from .api import lasso, solve


def lambda_max(dictionary, signal):
    """Smallest lambda for which the LASSO solution is all zeros."""
    return float(np.abs(np.asarray(dictionary).T @ np.asarray(signal)).max())


def find_lambda(dictionary, signal, criterion="bic", n_lambdas=30, min_ratio=1e-3,
                backend="fista", active_threshold=1e-3, **solver_kwargs):
    """
    Returns (best_lambda, best_result, table).

    table is a list of dicts, one per lambda tried, with the score, solution
    stats and x_refit (least-squares coefficients on the active set). LASSO
    shrinks coefficients by about lambda; x_refit undoes that if you want
    unbiased magnitudes once the support is chosen.
    """
    dictionary = np.asarray(dictionary, dtype=float)
    signal = np.asarray(signal, dtype=float).ravel()
    n_samples = len(signal)
    criterion = criterion.lower()
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")

    top = lambda_max(dictionary, signal)
    lambdas = np.geomspace(top, top * min_ratio, n_lambdas)

    best = (np.inf, None, None)
    table = []
    x_prev = None

    for lam in lambdas:
        result = solve(lasso(dictionary, signal, lam), backend=backend, x0=x_prev,
                       active_threshold=active_threshold, **solver_kwargs)
        x_prev = result.x_raw

        active = result.active
        n_active = len(active)
        x_refit = np.zeros(dictionary.shape[1])
        if n_active:
            x_refit[active], *_ = np.linalg.lstsq(dictionary[:, active], signal, rcond=None)
        rss = float(np.sum((signal - dictionary @ x_refit) ** 2))
        dof = n_samples - n_active
        if dof <= 0:
            score = np.inf                       # interpolating, no evidence left
        else:
            variance = rss / dof + 1e-12
            penalty = n_active * (np.log(n_samples) if criterion == "bic" else 2)
            score = n_samples * np.log(variance) + penalty

        table.append({"lambda": float(lam), "score": score, "rss": rss, "n_active": n_active,
                      "objective": result.objective, "x_refit": x_refit})
        if score < best[0]:
            best = (score, float(lam), result)

    return best[1], best[2], table
