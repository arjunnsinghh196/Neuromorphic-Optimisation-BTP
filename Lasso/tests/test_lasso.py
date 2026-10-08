"""
pytest Lasso/tests

Covers the normal path, the edge cases, and the error paths.
"""

import numpy as np
import pytest

import neuroopt as no

rng = np.random.default_rng(0)


def random_lasso(n_samples=20, n_atoms=30, n_nonzero=3, lam=0.1, noise=0.0):
    Phi = rng.standard_normal((n_samples, n_atoms))
    Phi /= np.linalg.norm(Phi, axis=0)
    x_true = np.zeros(n_atoms)
    # magnitudes in [0.5, 2] so every planted atom is clearly above any noise
    x_true[rng.choice(n_atoms, n_nonzero, replace=False)] = rng.choice([-1, 1], n_nonzero) * rng.uniform(0.5, 2, n_nonzero)
    s = Phi @ x_true + noise * rng.standard_normal(n_samples)
    return Phi, s, x_true, no.lasso(Phi, s, lam)


# ---------------------------------------------------------------- normal path
def test_all_backends_agree():
    _, _, _, problem = random_lasso()
    ref = no.solve(problem, backend="reference")
    for backend, tol, kw in [("ista", 1e-5, {}), ("fista", 1e-5, {}), ("analog", 1e-5, {}),
                             ("spiking", 5e-3, {"precision": 1e-3, "t_max": 1500})]:
        res = no.solve(problem, backend=backend, **kw)
        assert np.linalg.norm(res.x_raw - ref.x) < tol, backend
        # compare supports at a common threshold so the spiking noise floor doesn't decide
        assert np.array_equal(np.abs(res.x_raw) > 1e-2, np.abs(ref.x) > 1e-2), backend


def test_reference_is_optimal():
    _, _, _, problem = random_lasso()
    assert no.solve(problem, backend="reference").kkt_residual < 1e-6


def test_truncation_leaves_no_tiny_values():
    _, _, _, problem = random_lasso()
    res = no.solve(problem, backend="spiking")
    nonzero = res.x[res.x != 0]
    assert np.all(np.abs(nonzero) > res.active_threshold)
    assert res.x_raw.shape == res.x.shape


def test_spiking_is_deterministic():
    _, _, _, problem = random_lasso()
    a = no.solve(problem, backend="spiking", t_max=100)
    b = no.solve(problem, backend="spiking", t_max=100)
    assert np.array_equal(a.x_raw, b.x_raw)
    assert a.stats["total_spikes"] == b.stats["total_spikes"]


# ---------------------------------------------------------------- edge cases
def test_zero_signal_gives_zero_solution_and_no_spikes():
    Phi = rng.standard_normal((5, 8))
    Phi /= np.linalg.norm(Phi, axis=0)
    res = no.solve(no.lasso(Phi, np.zeros(5), 0.1), backend="spiking", t_max=50)
    assert np.all(res.x == 0)
    assert res.stats["total_spikes"] == 0


def test_lambda_at_or_above_lambda_max_gives_zero():
    Phi, s, _, _ = random_lasso()
    lam = no.lambda_max(Phi, s)
    for backend, kw in [("fista", {}), ("analog", {"t_max": 200}), ("spiking", {"t_max": 200})]:
        res = no.solve(no.lasso(Phi, s, lam * 1.01), backend=backend, **kw)
        assert np.all(res.x == 0), backend


def test_lambda_zero_is_least_squares():
    Phi = rng.standard_normal((6, 4))
    Phi /= np.linalg.norm(Phi, axis=0)
    s = rng.standard_normal(6)
    x_ls, *_ = np.linalg.lstsq(Phi, s, rcond=None)
    res = no.solve(no.lasso(Phi, s, 0.0), backend="analog", tol=1e-12)
    assert np.allclose(res.x, x_ls, atol=1e-5)


def test_orthonormal_dictionary_spiking_is_exact():
    # no lateral coupling -> no spike noise
    Phi, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    s = Phi @ np.array([1, 0, -0.5, 0, 0, 0.8, 0, 0])
    problem = no.lasso(Phi, s, 0.1)
    ref = no.solve(problem, backend="reference")
    res = no.solve(problem, backend="spiking", t_max=200)
    assert np.linalg.norm(res.x - ref.x) < 1e-8


def test_non_unit_norm_dictionary_is_rescaled():
    Phi = rng.standard_normal((10, 6)) * np.array([3, 0.5, 1, 2, 0.1, 1])
    s = rng.standard_normal(10)
    problem = no.lasso(Phi, s, 0.05)
    assert no.network_for(problem).rescaled
    ref = no.solve(problem, backend="reference")
    res = no.solve(problem, backend="spiking", precision=1e-3, t_max=1500)
    assert np.linalg.norm(res.x - ref.x) < 1e-2 * max(1, np.abs(ref.x).max())


def test_single_atom():
    Phi = np.ones((4, 1)) / 2
    s = np.array([1.0, 1.0, 1.0, 1.0])
    ref = no.solve(no.lasso(Phi, s, 0.1), backend="reference")
    res = no.solve(no.lasso(Phi, s, 0.1), backend="spiking", t_max=100)
    assert abs(res.x[0] - ref.x[0]) < 1e-6


def test_overcomplete_dictionary():
    _, _, x_true, problem = random_lasso(n_samples=10, n_atoms=40, n_nonzero=2, lam=0.05)
    ref = no.solve(problem, backend="reference")
    res = no.solve(problem, backend="spiking", t_max=1500)
    assert np.array_equal(res.active, ref.active)


def test_nonneg_lasso_has_no_negatives():
    Phi, s, _, _ = random_lasso()
    res = no.solve(no.nnls(Phi, s, 0.05), backend="spiking", t_max=500)
    assert np.all(res.x >= 0)
    assert res.violation == 0


def test_constrained_qp_all_backends():
    qp = no.quadratic_program(np.eye(3) * 2, [-1.0, -2.0, -3.0], matrix=np.ones((1, 3)), rhs=[1.0])
    ref = no.solve(qp, backend="reference")
    assert np.allclose(ref.x, [-1 / 6, 1 / 3, 5 / 6], atol=1e-6)
    for backend in ("analog", "spiking"):
        res = no.solve(qp, backend=backend, precision=1e-3) if backend == "spiking" else no.solve(qp, backend=backend)
        assert np.linalg.norm(res.x - ref.x) < 1e-2, backend
        assert res.violation < 5e-3, backend


def test_warm_start_converges_faster():
    _, _, _, problem = random_lasso()
    cold = no.solve(problem, backend="fista")
    warm = no.solve(problem, backend="fista", x0=cold.x)
    assert warm.iterations < cold.iterations


# ---------------------------------------------------------------- lambda tuning
def test_find_lambda_recovers_support_with_noise():
    Phi, s, x_true, _ = random_lasso(n_samples=60, n_atoms=20, n_nonzero=3, noise=0.05)
    lam, res, table = no.find_lambda(Phi, s, criterion="bic")
    assert 0 < lam < no.lambda_max(Phi, s)
    assert len(table) == 30
    # true atoms must all be found; a couple of small extras are normal with noise
    assert set(np.flatnonzero(x_true)) <= set(res.active)
    assert res.n_active <= 7


def test_find_lambda_rejects_bad_criterion():
    Phi, s, _, _ = random_lasso()
    with pytest.raises(ValueError):
        no.find_lambda(Phi, s, criterion="rmse")


# ---------------------------------------------------------------- error paths
def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        no.lasso(np.ones((3, 2)), np.ones(4), 0.1).canonicalize()


def test_negative_lambda_raises():
    with pytest.raises(ValueError):
        no.lasso(np.eye(3), np.ones(3), -0.1).canonicalize()


def test_non_psd_gram_raises():
    with pytest.raises(ValueError):
        no.Quadratic([[1, 0], [0, -1]])


def test_ista_rejects_constraints():
    qp = no.quadratic_program(np.eye(2), [1.0, 1.0], matrix=np.ones((1, 2)), rhs=[1.0])
    with pytest.raises(ValueError):
        no.solve(qp, backend="ista")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        no.solve(no.lasso(np.eye(2), np.ones(2), 0.1), backend="quantum")


def test_describe_runs():
    Phi, s, _, _ = random_lasso()
    text = no.describe(no.lasso(Phi, s, 0.1))
    assert "integrate-and-fire with dead zone" in text
