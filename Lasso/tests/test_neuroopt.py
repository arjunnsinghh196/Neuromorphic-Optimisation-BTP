"""
Validation suite.

These are not smoke tests -- each one checks a specific claim from the
derivation, including the three places where the original notes were wrong.
"""

import numpy as np
import pytest

import neuroopt as no
from neuroopt.compile import compile_network
from neuroopt.objectives import CanonicalProblem


def make_sparse_problem(M=40, N=80, k=4, lam=0.05, seed=0, coherence=0.0):
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((M, N))
    if coherence:  # make atoms correlated -> ill-conditioned Gram
        Phi = (1 - coherence) * Phi + coherence * rng.standard_normal((M, 1))
    Phi /= np.linalg.norm(Phi, axis=0)
    a = np.zeros(N)
    a[rng.choice(N, k, replace=False)] = rng.standard_normal(k) * 2
    s = Phi @ a
    return Phi, s, a, lam


# --------------------------------------------------------------------------- #
# 1. the three solvers agree
# --------------------------------------------------------------------------- #
def test_ista_matches_reference():
    Phi, s, _, lam = make_sparse_problem()
    prob = no.lasso(Phi, s, lam)
    ref = no.solve(prob, backend="reference")
    ista = no.solve(prob, backend="fista", max_iter=20000, tol=1e-14)
    assert np.linalg.norm(ista.x - ref.x) < 1e-6
    assert ista.kkt_residual < 1e-7


def test_analog_lca_matches_reference():
    Phi, s, _, lam = make_sparse_problem()
    prob = no.lasso(Phi, s, lam)
    ref = no.solve(prob, backend="reference")
    lca = no.solve(prob, backend="analog", t_max=400, tol=1e-13)
    assert np.linalg.norm(lca.x - ref.x) < 1e-6


def test_spiking_matches_reference():
    Phi, s, _, lam = make_sparse_problem()
    prob = no.lasso(Phi, s, lam)
    ref = no.solve(prob, backend="reference")
    spk = no.solve(prob, backend="spiking", precision=1e-3)
    assert np.linalg.norm(spk.x - ref.x) < 1e-4
    assert spk.stats["active_fraction"] < 0.2
    assert set(spk.support) == set(ref.support)


# --------------------------------------------------------------------------- #
# 2. ISTA IS forward Euler on the LCA flow, step size 1
# --------------------------------------------------------------------------- #
def test_ista_is_unit_step_forward_euler():
    """One ISTA step must equal one unit-step Euler step of u_dot = -u + Wa + b.

    Tall dictionary so that lambda_max(Phi^T Phi) < 2 and the unit-step
    iteration is in its stable regime; the identity itself holds regardless.
    """
    Phi, s, _, lam = make_sparse_problem(M=200, N=12, k=3)
    prob = no.lasso(Phi, s, lam).canonicalize()
    spec = compile_network(prob)
    assert np.allclose(np.diag(prob.Q), 1.0)
    assert np.linalg.eigvalsh(prob.Q).max() < 2.0

    u = np.zeros(prob.n)
    a = np.zeros(prob.n)
    for _ in range(100):
        u = u + 1.0 * (-u + spec.W @ a + spec.bias)      # forward Euler, eta = 1
        a_euler = spec.activation(u)
        v = a - (prob.Q @ a + prob.c)                     # ISTA gradient step
        a_ista = np.sign(v) * np.maximum(np.abs(v) - prob.lam1, 0.0)
        assert np.allclose(a_euler, a_ista, atol=1e-12)
        a = a_euler
    assert prob.kkt_residual(a) < 1e-8


# --------------------------------------------------------------------------- #
# 3. BUG 1: the self-synapse must be absent
# --------------------------------------------------------------------------- #
def test_self_synapse_would_change_the_problem():
    """Including w_ii = 1 turns LASSO into an elastic net with lam2 = 1/2.

    We solve with the correct (zero-diagonal) network, then with the diagonal
    left in, and check that the second one matches the *elastic net* solution.
    """
    Phi, s, _, lam = make_sparse_problem(M=25, N=40, lam=0.08)
    prob = no.lasso(Phi, s, lam).canonicalize()
    spec = compile_network(prob)

    correct = no.solve(prob, backend="analog", t_max=600, tol=1e-13)

    # deliberately reintroduce the self-synapse: W = I - Q with diagonal kept
    bad = compile_network(prob)
    bad.W = np.eye(prob.n) - prob.Q          # diag(W) = 0 already since diag(Q)=1
    bad.W = -(prob.Q)                        # = W_syn with w_ii = 1 folded in
    from neuroopt.solvers import solve_analog
    wrong = solve_analog(prob, spec=bad, t_max=600, tol=1e-13)

    en = no.elastic_net(Phi, s, lam, 0.5)
    en_ref = no.solve(en, backend="reference")

    assert np.linalg.norm(correct.x - no.solve(prob, backend="reference").x) < 1e-6
    assert np.linalg.norm(wrong.x - en_ref.x) < 1e-4
    assert np.linalg.norm(wrong.x - correct.x) > 1e-2   # materially different


# --------------------------------------------------------------------------- #
# 4. elastic net: nu_f = 2*lam2 + 1 reproduces the proximal operator
# --------------------------------------------------------------------------- #
def test_elastic_net_threshold_mapping():
    Phi, s, _, _ = make_sparse_problem(M=30, N=50)
    lam1, lam2 = 0.06, 0.4
    prob = no.elastic_net(Phi, s, lam1, lam2)
    spec = compile_network(prob.canonicalize())
    assert np.allclose(spec.nu_f, 2 * lam2 + 1)
    assert np.allclose(spec.lam1, lam1)

    u = np.linspace(-2, 2, 401)
    expected = np.sign(u) * np.maximum(np.abs(u) - lam1, 0.0) / (2 * lam2 + 1)
    got = spec.activation(np.broadcast_to(u[:, None], (401, spec.n)).copy())
    assert np.allclose(got[:, 0], expected)

    ref = no.solve(prob, backend="reference")
    prec = 1e-3
    spk = no.solve(prob, backend="spiking", t_max=150, precision=prec)
    # assert the contract the solver actually promises: `prec` relative to the
    # largest coefficient.  A fixed absolute bound would only pass by accident,
    # as it did while the auto-tuner was over-provisioning gamma by ~50x.
    assert np.linalg.norm(spk.x - ref.x) < prec * np.abs(ref.x).max()


def test_l2_reduces_condition_number():
    Phi, s, _, _ = make_sparse_problem(coherence=0.9)
    k0 = no.conditioning_report(no.lasso(Phi, s, 0.05))["condition_number"]
    k1 = no.conditioning_report(no.elastic_net(Phi, s, 0.05, 0.25))["condition_number"]
    assert k1 < k0


# --------------------------------------------------------------------------- #
# 5. normalisation: unnormalised atoms must not silently change the threshold
# --------------------------------------------------------------------------- #
def test_unnormalized_dictionary_is_rescaled_correctly():
    rng = np.random.default_rng(3)
    M, N = 25, 40
    Phi = rng.standard_normal((M, N)) * rng.uniform(0.3, 3.0, N)  # varied norms
    a = np.zeros(N); a[[1, 9, 20]] = [1.5, -2.0, 0.7]
    s = Phi @ a
    prob = no.lasso(Phi, s, 0.05)
    spec = compile_network(prob.canonicalize())
    assert spec.normalized
    assert np.allclose(np.diag(spec.Q), 1.0)
    ref = no.solve(prob, backend="reference")
    spk = no.solve(prob, backend="spiking", t_max=1500, precision=1e-3)
    assert np.linalg.norm(spk.x - ref.x) < 1e-4


# --------------------------------------------------------------------------- #
# 6. BUG 2/3: constrained QP, projected duals and the rectified integral
# --------------------------------------------------------------------------- #
def test_constrained_qp_matches_reference():
    rng = np.random.default_rng(5)
    n = 6
    B = rng.standard_normal((n, n))
    Q = B @ B.T + 0.5 * np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((4, n))
    k = A @ rng.standard_normal(n) * 0.3 - 0.2
    prob = no.quadratic_program(Q, c, A, k)
    ref = no.solve(prob, backend="reference")
    spk = no.solve(prob, backend="spiking", t_max=600, precision=1e-3)
    assert spk.violation < 1e-6
    assert abs(spk.objective - ref.objective) < 1e-4 * max(1, abs(ref.objective))
    assert np.linalg.norm(spk.x - ref.x) < 1e-3


def test_multiplier_can_return_to_zero():
    """A transiently violated constraint must be able to go inactive again.

    With dy/dt = [Ax-b]_+ every multiplier is monotone non-decreasing, so this
    test fails; with the projected dynamics w_dot = beta(Ax-k), v = [w]_+ it
    passes.  The trajectory starts at x = 0, overshoots a constraint that is
    inactive at the optimum, and must recover.
    """
    Q = np.array([[2.0, 0.0], [0.0, 2.0]])
    c = np.array([-1.0, -1.0])            # unconstrained optimum at (0.5, 0.5)
    A = np.array([[1.0, 1.0]])
    k = np.array([5.0])                   # slack at the optimum
    prob = no.quadratic_program(Q, c, A, k)
    res = no.solve(prob, backend="analog", t_max=200)
    assert np.allclose(res.x, [0.5, 0.5], atol=1e-6)
    assert res.y is not None and res.y[0] < 1e-9   # multiplier returned to zero


def test_sparse_constrained_qp():
    """l1 + linear inequalities at once: dead-zone neurons plus a dual layer."""
    rng = np.random.default_rng(11)
    n = 12
    B = rng.standard_normal((n, n))
    Q = B @ B.T / n + 0.3 * np.eye(n)
    c = rng.standard_normal(n)
    A = np.ones((1, n))
    k = np.array([0.5])
    prob = no.sparse_qp(Q, c, lam1=0.15, A=A, k=k)
    ref = no.solve(prob, backend="reference")
    spk = no.solve(prob, backend="spiking", t_max=900, precision=1e-4)
    assert spk.violation < 1e-6
    assert np.linalg.norm(spk.x - ref.x) < 1e-4


def test_nonnegative_neurons():
    Phi, s, _, lam = make_sparse_problem(M=30, N=45, lam=0.05, seed=7)
    prob = no.nnls(Phi, s, lam)
    ref = no.solve(prob, backend="reference")
    prec = 1e-3
    spk = no.solve(prob, backend="spiking", t_max=150, precision=prec)
    assert np.all(spk.x >= -1e-12)
    assert np.linalg.norm(spk.x - ref.x) < prec * np.abs(ref.x).max()


# --------------------------------------------------------------------------- #
# 7. rate readout converges as O(1/T)
# --------------------------------------------------------------------------- #
def test_rate_readout_converges_as_one_over_T():
    """The pure rate code carries an O(1/T) transient bias the analog readout has not."""
    Phi, s, _, lam = make_sparse_problem(M=30, N=50, lam=0.06)
    prob = no.lasso(Phi, s, lam)
    errs = []
    for T in (50.0, 100.0, 200.0, 400.0):
        r = no.solve(prob, backend="spiking", t_max=T, readout="rate",
                     spike_resolution=400, tol=0.0)
        errs.append(r.stats["readout_rate_vs_analog"])
    assert errs[1] < errs[0] and errs[2] < errs[1] and errs[3] < errs[2]
    ratio = errs[0] / errs[3]           # 8x more time
    assert 3.0 < ratio < 20.0           # consistent with ~1/T, not 1/sqrt(T)


def test_accuracy_improves_with_spike_resolution():
    """Error falls with the value-per-spike; spike count grows linearly."""
    Phi, s, _, lam = make_sparse_problem(M=30, N=50, lam=0.06)
    prob = no.lasso(Phi, s, lam)
    ref = no.solve(prob, backend="reference")
    errs, spikes = [], []
    for g in (25, 100, 400):
        r = no.solve(prob, backend="spiking", t_max=60, spike_resolution=g, tol=0.0)
        errs.append(np.linalg.norm(r.x - ref.x))
        spikes.append(r.stats["total_spikes"])
    assert errs[0] > errs[1] > errs[2]
    assert spikes[0] < spikes[1] < spikes[2]


# --------------------------------------------------------------------------- #
# 8. sparsity, spikes and energy move together
# --------------------------------------------------------------------------- #
def test_higher_lambda_gives_fewer_spikes():
    Phi, s, _, _ = make_sparse_problem()
    counts = []
    for lam in (0.02, 0.1, 0.3):
        r = no.solve(no.lasso(Phi, s, lam), backend="spiking", t_max=60,
                     spike_resolution=200, tol=0.0)
        counts.append(r.stats["total_spikes"])
    assert counts[0] > counts[1] > counts[2]


def test_ridge_is_dense_and_flagged():
    Phi, s, _, _ = make_sparse_problem()
    txt = no.describe(no.ridge(Phi, s, 0.1))
    assert "poor" in txt
    r = no.solve(no.ridge(Phi, s, 0.1), backend="analog", t_max=200)
    assert r.sparsity < 0.05          # essentially no zeros


# --------------------------------------------------------------------------- #
# 9. guard rails
# --------------------------------------------------------------------------- #
def test_nonconvex_rejected():
    with pytest.raises(ValueError, match="positive semi-definite"):
        no.quadratic_program(np.array([[1.0, 0.0], [0.0, -1.0]]), np.zeros(2))


def test_ista_rejects_constraints():
    prob = no.quadratic_program(np.eye(2), np.ones(2), np.ones((1, 2)), np.array([1.0]))
    with pytest.raises(ValueError, match="constrained"):
        no.solve(prob, backend="ista")


def test_describe_runs():
    Phi, s, _, lam = make_sparse_problem()
    txt = no.describe(no.lasso(Phi, s, lam))
    assert "integrate-and-fire" in txt and "nu_f" in txt
