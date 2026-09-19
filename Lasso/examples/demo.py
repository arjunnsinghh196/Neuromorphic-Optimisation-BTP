"""End-to-end demo: objective in, neuron model out, solution on a spiking network."""

import numpy as np

import neuroopt as no


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def sparse_coding_data(M=48, N=96, k=5, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal((M, N))
    Phi /= np.linalg.norm(Phi, axis=0)          # unit atoms: required for the mapping
    a = np.zeros(N)
    a[rng.choice(N, k, replace=False)] = rng.standard_normal(k) * 2
    return Phi, Phi @ a + noise * rng.standard_normal(M), a


# --------------------------------------------------------------------------- #
banner("1. LASSO -- the objective compiles to a neuron model")
Phi, s, a_true = sparse_coding_data()
prob = no.lasso(Phi, s, lam=0.08)
print(no.describe(prob))

banner("2. Four realisations of the same dynamical system")
no.compare(prob)

banner("3. Elastic net: l2 moves the firing threshold, nothing else")
en = no.elastic_net(Phi, s, lam1=0.08, lam2=0.3)
spec_l1 = no.network_for(no.lasso(Phi, s, 0.08))
spec_en = no.network_for(en)
print(f"weights identical           : {np.allclose(spec_l1.W, spec_en.W)}")
print(f"dead zone   lam1            : {spec_l1.lam1[0]:.3f} -> {spec_en.lam1[0]:.3f}")
print(f"threshold   nu_f = 2*lam2+1 : {spec_l1.nu_f[0]:.3f} -> {spec_en.nu_f[0]:.3f}")
print(f"gain above threshold        : {1/spec_l1.nu_f[0]:.3f} -> {1/spec_en.nu_f[0]:.3f}")

banner("4. Constrained QP -- two layers, silent while constraints are slack")
rng = np.random.default_rng(7)
n, m = 10, 6
B = rng.standard_normal((n, n))
Q = B @ B.T / n + 0.4 * np.eye(n)
c = rng.standard_normal(n)
A = rng.standard_normal((m, n))
k = A @ rng.standard_normal(n) * 0.4
qp = no.quadratic_program(Q, c, A, k)
ref = no.solve(qp, backend="reference")
spk = no.solve(qp, backend="spiking", t_max=1200, precision=1e-3)
print(f"|x - x_ref|        : {np.linalg.norm(spk.x - ref.x):.3e}")
print(f"max violation      : {spk.violation:.3e}")
print(f"active constraints : {int(np.sum(spk.y > 1e-8))} of {m}")
print(f"multipliers        : {np.array2string(spk.y, precision=4)}")
print(f"reference          : {np.array2string(ref.y, precision=4)}")
print("\nThe zeros are constraint neurons that never emitted a single spike.")

banner("5. Precision costs spikes: error ~ 1/gamma")
ref2 = no.solve(prob, backend="reference")
gs = np.array([100, 300, 1000, 3000, 10000])
errs = []
print(f"{'gamma':>10}{'|x-ref|':>12}{'spikes':>12}{'nnz':>6}")
for g in gs:
    r = no.solve(prob, backend="spiking", t_max=200, spike_resolution=int(g), tol=0.0)
    errs.append(np.linalg.norm(r.x - ref2.x))
    print(f"{g:>10}{errs[-1]:>12.2e}{r.stats['total_spikes']:>12.4g}{len(r.support):>6}")
slope = np.polyfit(np.log(gs), np.log(errs), 1)[0]
print(f"\nfitted exponent: {slope:.3f}   (shot noise would give -0.5)")
print("The subtractive reset makes each neuron a sigma-delta modulator.")

banner("6. Sparsity is the whole point: lambda controls the spike budget")
print(f"{'lambda':>10}{'nnz':>7}{'spikes':>12}{'syn events':>14}{'objective':>14}")
for lam in (0.02, 0.05, 0.1, 0.3):
    r = no.solve(no.lasso(Phi, s, lam), backend="spiking", t_max=120,
                 spike_resolution=250, tol=0.0)
    print(f"{lam:>10}{len(r.support):>7}{r.stats['total_spikes']:>12.4g}"
          f"{r.stats['synaptic_events']:>14.4g}{r.objective:>14.6g}")

banner("7. Ridge: convex, easy, and a bad fit for event-driven dynamics")
rg = no.solve(no.ridge(Phi, s, 0.3), backend="spiking", t_max=120,
              spike_resolution=250, tol=0.0)
ls = no.solve(no.lasso(Phi, s, 0.08), backend="spiking", t_max=120,
              spike_resolution=250, tol=0.0)
for tag, rr in (("ridge", rg), ("lasso", ls)):
    print(f"{tag:<6}: {len(rr.support):>3} active, "
          f"{rr.stats['total_spikes']:>10.4g} spikes, "
          f"{rr.stats['synaptic_events']:>11.4g} synaptic events")

banner("8. Where the spiking error comes from: a controlled experiment")
r3 = np.random.default_rng(1)
Phi_o = np.linalg.qr(r3.standard_normal((48, 48)))[0]       # orthonormal -> W = 0
a_o = np.zeros(48)
a_o[r3.choice(48, 5, replace=False)] = r3.standard_normal(5) * 2
prob_o = no.lasso(Phi_o, Phi_o @ a_o, lam=0.08)
ref_o = no.solve(prob_o, backend="reference")
for g in (100, 1000, 10000):
    e = np.linalg.norm(no.solve(prob_o, backend="spiking", t_max=200,
                                spike_resolution=g, tol=0.0).x - ref_o.x)
    print(f"orthonormal dictionary, gamma={g:>6}:  err = {e:.3e}")
an = no.solve(prob_o, backend="analog", t_max=200, tol=0.0)
print(f"analog (no spikes at all)          :  err = {np.linalg.norm(an.x-ref_o.x):.3e}")
print("\nRemove the lateral coupling and the spike error vanishes entirely:")
print("the cost comes from neurons receiving each other's quantisation residuals,")
print("not from a neuron quantising its own output.")

banner("9. Conditioning: what the continuous dynamics do and do not fix")
Phi_c = 0.35 * Phi + 0.65 * rng.standard_normal((Phi.shape[0], 1))
Phi_c /= np.linalg.norm(Phi_c, axis=0)
s_c = Phi_c @ a_true
for name, p in [("well conditioned", no.lasso(Phi, s, 0.08)),
                ("coherent dictionary", no.lasso(Phi_c, s_c, 0.08)),
                ("coherent + l2", no.elastic_net(Phi_c, s_c, 0.08, 0.5))]:
    rep = no.conditioning_report(p)
    lmr = rep["lambda_min_on_support"]
    print(f"{name:<22} lambda_max={rep['lambda_max']:>7.3g}  "
          f"lambda_min={rep['lambda_min']:>10.3g}  "
          f"on support={0.0 if lmr is None else lmr:>7.3g} "
          f"(|supp|={rep['support_size']})")
