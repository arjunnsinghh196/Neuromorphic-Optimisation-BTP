"""
Presentation figures.

Six plots, in the order you would present them.  Each one is a separate cell in
the notebook version; run this file directly to write all six as PNGs.

    python make_figures.py          # writes figures/*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import neuroopt as no

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


# --------------------------------------------------------------------------- #
# shared problem
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
M, N, K = 48, 96, 5
Phi = rng.standard_normal((M, N))
Phi /= np.linalg.norm(Phi, axis=0)
a_true = np.zeros(N)
a_true[rng.choice(N, K, replace=False)] = rng.standard_normal(K) * 2
s = Phi @ a_true + 0.01 * rng.standard_normal(M)

prob = no.lasso(Phi, s, lam=0.08)
ref = no.solve(prob, backend="reference")
print(f"reference: objective={ref.objective:.6f}  nonzeros={len(ref.support)}")


# --------------------------------------------------------------------------- #
# FIGURE 1 -- it works: the network recovers the planted sparse signal
# --------------------------------------------------------------------------- #
def fig1():
    spk = no.solve(prob, backend="spiking", precision=1e-3)
    idx = np.arange(N)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.axhline(0, color="0.7", lw=0.8)
    ax.vlines(idx, 0, a_true, color="C0", lw=2.4, alpha=0.85,
              label="planted coefficients (the truth)")
    ax.plot(idx, spk.x, "o", ms=5, color="C1", mfc="none", mew=1.4,
            label="recovered by the spiking network")

    ax.set(xlabel="dictionary atom index", ylabel="coefficient value",
           title=f"96 atoms, 5 planted non-zeros -- network found "
                 f"{len(spk.support)} active, error {np.linalg.norm(spk.x-ref.x):.1e}")
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(x=0.01)
    plt.tight_layout(); plt.savefig(f"{OUT}/1_recovery.png"); plt.close()
    print("fig 1 done")


# --------------------------------------------------------------------------- #
# FIGURE 2 -- the network settles, and most neurons stay silent
# --------------------------------------------------------------------------- #
def fig2():
    an = no.solve(prob, backend="analog", t_max=120, tol=0.0, record_every=2)
    sp = no.solve(prob, backend="spiking", t_max=120, tol=0.0,
                  spike_resolution=1000, record_every=20)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    gap_a = np.abs(np.array(an.history["objective"]) - ref.objective) + 1e-17
    gap_s = np.abs(np.array(sp.history["objective"]) - ref.objective) + 1e-17
    ax[0].semilogy(an.history["time"], gap_a, label="analog (continuous)")
    ax[0].semilogy(sp.history["time"], gap_s, label="spiking")
    ax[0].set(xlabel="time", ylabel="objective gap",
              title="Settling: the answer is where it stops")
    ax[0].legend()

    ax[1].semilogy(an.history["time"], an.history["kkt_residual"], label="analog")
    ax[1].semilogy(sp.history["time"], sp.history["kkt_residual"], label="spiking")
    ax[1].set(xlabel="time", ylabel="KKT residual",
              title="Optimality conditions being satisfied")
    ax[1].legend()

    ax[2].plot(sp.history["time"], np.array(sp.history["spikes"]) / 1e3)
    ax[2].set(xlabel="time", ylabel="cumulative spikes (thousands)",
              title="Spike economy: steep, then flat")

    plt.tight_layout(); plt.savefig(f"{OUT}/2_convergence.png"); plt.close()
    print("fig 2 done")


# --------------------------------------------------------------------------- #
# FIGURE 3 -- the main scientific result: error ~ 1/gamma, not 1/sqrt(gamma)
# --------------------------------------------------------------------------- #
def fig3():
    gs = np.array([100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000])
    errs, spikes = [], []
    for g in gs:
        r = no.solve(prob, backend="spiking", t_max=200,
                     spike_resolution=int(g), tol=0.0)
        errs.append(np.linalg.norm(r.x - ref.x))
        spikes.append(r.stats["total_spikes"])
    errs, spikes = np.array(errs), np.array(spikes)
    slope = np.polyfit(np.log(gs), np.log(errs), 1)[0]
    C = np.median(errs * gs)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].loglog(gs, errs, "o-", lw=1.6, label=f"measured (slope {slope:.3f})")
    ax[0].loglog(gs, C / gs, "r--", lw=1.4, label=r"$1/\gamma$  (sigma-delta)")
    ax[0].loglog(gs, errs[0] * (gs / gs[0]) ** -0.5, "k--", lw=1.4,
                 label=r"$1/\sqrt{\gamma}$  (shot noise -- what I predicted)")
    ax[0].set(xlabel=r"$\gamma$   (value carried per spike)",
              ylabel=r"$\|x - x^*\|$",
              title="Accuracy follows the sigma-delta law")
    ax[0].legend(fontsize=8.5)

    ax[1].loglog(spikes, errs, "o-", lw=1.6, color="C2")
    ax[1].set(xlabel="total spikes emitted", ylabel=r"$\|x - x^*\|$",
              title="What precision actually costs")
    for g, x, y in zip(gs, spikes, errs):
        if g in (100, 1000, 10000):
            ax[1].annotate(rf"$\gamma$={g}", (x, y), textcoords="offset points",
                           xytext=(6, 6), fontsize=8)

    plt.tight_layout(); plt.savefig(f"{OUT}/3_gamma_law.png"); plt.close()
    print(f"fig 3 done   slope={slope:.3f}  C={C:.4g}")
    return slope, C


# --------------------------------------------------------------------------- #
# FIGURE 4 -- the error is deterministic and chaotic in gamma
# --------------------------------------------------------------------------- #
def fig4(C):
    gs = np.arange(200, 501, 5)
    rel = np.array([
        np.linalg.norm(no.solve(prob, backend="spiking", t_max=200,
                                spike_resolution=int(g), tol=0.0).x - ref.x) * g / C
        for g in gs])

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(gs, rel, ".-", lw=1.1, ms=5)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label=r"mean of the $1/\gamma$ law")
    ax.fill_between(gs, 0, rel, alpha=0.12)
    ax.set(xlabel=r"$\gamma$", ylabel=r"error $\times\,\gamma\,/\,C$",
           title="The error is NOT smooth in the spike budget: "
                 "you cannot tune $\\gamma$ by trial and error")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/4_fine_structure.png"); plt.close()
    print(f"fig 4 done   spread {rel.min():.2f} to {rel.max():.2f}")


# --------------------------------------------------------------------------- #
# FIGURE 5 -- the clincher: remove the coupling and the error disappears
# --------------------------------------------------------------------------- #
def fig5():
    r2 = np.random.default_rng(1)
    Phi_o = np.linalg.qr(r2.standard_normal((48, 48)))[0]      # orthonormal -> W = 0
    a_o = np.zeros(48)
    a_o[r2.choice(48, 5, replace=False)] = r2.standard_normal(5) * 2
    prob_o = no.lasso(Phi_o, Phi_o @ a_o, lam=0.08)
    ref_o = no.solve(prob_o, backend="reference")

    gs = np.array([100, 300, 1000, 3000, 10000])
    e_coupled = np.array([np.linalg.norm(
        no.solve(prob, backend="spiking", t_max=200,
                 spike_resolution=int(g), tol=0.0).x - ref.x) for g in gs])
    e_decoupled = np.array([np.linalg.norm(
        no.solve(prob_o, backend="spiking", t_max=200,
                 spike_resolution=int(g), tol=0.0).x - ref_o.x) for g in gs])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.loglog(gs, e_coupled, "o-", lw=1.8,
              label="overcomplete dictionary  (neurons inhibit each other)")
    ax.loglog(gs, e_decoupled, "s-", lw=1.8,
              label=r"orthonormal dictionary  ($W = 0$, no coupling)")
    ax.axhline(2.2e-16, color="k", ls=":", lw=1.2, label="double-precision epsilon")
    ax.set(xlabel=r"$\gamma$   (value carried per spike)", ylabel=r"$\|x - x^*\|$",
           title="Where the spiking error actually comes from")
    ax.legend(fontsize=8.5, loc="center left")
    plt.tight_layout(); plt.savefig(f"{OUT}/5_coupling_control.png"); plt.close()
    print(f"fig 5 done   decoupled errors: {e_decoupled}")


# --------------------------------------------------------------------------- #
# FIGURE 6 -- why sparsity is the point: lambda controls spikes and energy
# --------------------------------------------------------------------------- #
def fig6():
    lams = np.array([0.02, 0.04, 0.08, 0.15, 0.30, 0.50])
    nnz, spk = [], []
    for lam in lams:
        r = no.solve(no.lasso(Phi, s, lam), backend="spiking", t_max=120,
                     spike_resolution=1000, tol=0.0)
        nnz.append(len(r.support))
        spk.append(r.stats["total_spikes"] / 1e6)

    rr = no.solve(no.ridge(Phi, s, 0.3), backend="spiking", t_max=120,
                  spike_resolution=1000, tol=0.0)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(lams, nnz, "o-", lw=1.8)
    ax[0].set(xscale="log", xlabel=r"$\lambda$  (sparsity penalty)",
              ylabel="non-zero coefficients",
              title="Stronger penalty, fewer active neurons")

    ax[1].plot(lams, spk, "o-", lw=1.8, label="LASSO (sparse)")
    ax[1].axhline(rr.stats["total_spikes"] / 1e6, color="C3", ls="--", lw=1.8,
                  label=f"Ridge (dense, {len(rr.support)} active)")
    ax[1].set(xscale="log", xlabel=r"$\lambda$", ylabel="spikes (millions)",
              title="Sparse answers mean quiet networks")
    ax[1].legend(fontsize=9)

    plt.tight_layout(); plt.savefig(f"{OUT}/6_sparsity_spikes.png"); plt.close()
    print(f"fig 6 done   nnz={nnz}")


if __name__ == "__main__":
    fig1()
    fig2()
    slope, C = fig3()
    fig4(C)
    fig5()
    fig6()
    print(f"\nall figures written to {OUT}/")
