"""
Presentation figures. Run directly to write figures/*.png.

    python examples/make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import neuroopt as no

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


# --------------------------------------------------------------------------- #
# shared problem: 48 samples, 96 atoms, 5 planted non-zeros, a little noise
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
n_samples, n_atoms, n_planted = 48, 96, 5
dictionary = rng.standard_normal((n_samples, n_atoms))
dictionary /= np.linalg.norm(dictionary, axis=0)
x_true = np.zeros(n_atoms)
x_true[rng.choice(n_atoms, n_planted, replace=False)] = rng.choice([-1, 1], n_planted) * rng.uniform(0.8, 2.0, n_planted)
signal = dictionary @ x_true + 0.01 * rng.standard_normal(n_samples)

lam, _, sweep = no.find_lambda(dictionary, signal, criterion="bic")
problem = no.lasso(dictionary, signal, lam)
reference = no.solve(problem, backend="reference")
print(f"lambda (BIC) = {lam:.4f}   reference: objective={reference.objective:.6f}  active={reference.n_active}")


def spiking_error(prob, ref, spikes_per_unit, t_max=200):
    """||x - x_ref|| for a fixed-length spiking run at a given resolution."""
    result = no.solve(prob, backend="spiking", t_max=t_max, spikes_per_unit=spikes_per_unit, tol=None)
    return np.linalg.norm(result.x_raw - ref.x), result


# --------------------------------------------------------------------------- #
# 0 -- choosing lambda: the BIC sweep
# --------------------------------------------------------------------------- #
def fig0():
    lams = np.array([row["lambda"] for row in sweep])
    scores = np.array([row["score"] for row in sweep])
    n_active = np.array([row["n_active"] for row in sweep])
    finite = np.isfinite(scores)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].semilogx(lams[finite], scores[finite], "o-", lw=1.6)
    ax[0].axvline(lam, color="C3", ls="--", lw=1.4, label=f"chosen λ = {lam:.3g}")
    ax[0].set(xlabel="λ", ylabel="BIC score (lower is better)", title="Picking λ with BIC")
    ax[0].legend()

    ax[1].semilogx(lams, n_active, "o-", lw=1.6, color="C2")
    ax[1].axhline(n_planted, color="k", ls=":", lw=1.2, label=f"{n_planted} planted")
    ax[1].axvline(lam, color="C3", ls="--", lw=1.4)
    ax[1].set(xlabel="λ", ylabel="active atoms", title="Smaller λ, more atoms")
    ax[1].legend()

    plt.tight_layout(); plt.savefig(f"{OUT}/0_lambda_sweep.png"); plt.close()
    print("fig 0 done")


# --------------------------------------------------------------------------- #
# 1 -- it works: the network recovers the planted sparse signal
# --------------------------------------------------------------------------- #
def fig1():
    spiking = no.solve(problem, backend="spiking", precision=1e-3)
    index = np.arange(n_atoms)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.vlines(index, 0, x_true, color="C0", lw=2.4, alpha=0.85, label="planted coefficients (truth)")
    ax.plot(index, spiking.x, "o", ms=5, color="C1", mfc="none", mew=1.4, label="spiking network output")
    ax.set(xlabel="dictionary atom index", ylabel="coefficient",
           title=f"{n_atoms} atoms, {n_planted} planted — network found {spiking.n_active} active, "
                 f"error vs exact {np.linalg.norm(spiking.x - reference.x):.1e}")
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(x=0.01)
    plt.tight_layout(); plt.savefig(f"{OUT}/1_recovery.png"); plt.close()
    print("fig 1 done")


# --------------------------------------------------------------------------- #
# 2 -- the network settles, and most neurons stay silent
# --------------------------------------------------------------------------- #
def fig2():
    analog = no.solve(problem, backend="analog", t_max=120, tol=None, record_every=2)
    spiking = no.solve(problem, backend="spiking", t_max=120, tol=None, spikes_per_unit=1000, record_every=20)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    gap_analog = np.abs(np.array(analog.history["objective"]) - reference.objective) + 1e-17
    gap_spiking = np.abs(np.array(spiking.history["objective"]) - reference.objective) + 1e-17
    ax[0].semilogy(analog.history["time"], gap_analog, label="analog")
    ax[0].semilogy(spiking.history["time"], gap_spiking, label="spiking")
    ax[0].set(xlabel="time", ylabel="objective gap", title="Objective settles")
    ax[0].legend()

    ax[1].semilogy(analog.history["time"], analog.history["kkt_residual"], label="analog")
    ax[1].semilogy(spiking.history["time"], spiking.history["kkt_residual"], label="spiking")
    ax[1].set(xlabel="time", ylabel="KKT residual", title="Optimality conditions met")
    ax[1].legend()

    ax[2].plot(spiking.history["time"], np.array(spiking.history["spikes"]) / 1e3)
    ax[2].set(xlabel="time", ylabel="cumulative spikes (thousands)", title="Spikes: steep, then flat")

    plt.tight_layout(); plt.savefig(f"{OUT}/2_convergence.png"); plt.close()
    print("fig 2 done")


# --------------------------------------------------------------------------- #
# 3 -- error ~ 1/gamma, not 1/sqrt(gamma)
# --------------------------------------------------------------------------- #
def fig3():
    gammas = np.array([100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000])
    errors, spikes = [], []
    for gamma in gammas:
        err, result = spiking_error(problem, reference, int(gamma))
        errors.append(err)
        spikes.append(result.stats["total_spikes"])
    errors, spikes = np.array(errors), np.array(spikes)
    slope = np.polyfit(np.log(gammas), np.log(errors), 1)[0]
    scale = np.median(errors * gammas)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(gammas, errors, "o-", lw=1.6, label=f"measured (slope {slope:.2f})")
    ax[0].loglog(gammas, scale / gammas, "r--", lw=1.4, label=r"$1/\gamma$ (sigma-delta)")
    ax[0].loglog(gammas, errors[0] * (gammas / gammas[0]) ** -0.5, "k--", lw=1.4, label=r"$1/\sqrt{\gamma}$ (shot noise)")
    ax[0].set(xlabel=r"$\gamma$ (spikes per unit)", ylabel=r"$\|x - x^*\|$", title="Error follows the sigma-delta law")
    ax[0].legend(fontsize=8.5)

    ax[1].loglog(spikes, errors, "o-", lw=1.6, color="C2")
    ax[1].set(xlabel="total spikes", ylabel=r"$\|x - x^*\|$", title="What precision costs")
    for gamma, x, y in zip(gammas, spikes, errors):
        if gamma in (100, 1000, 10000):
            ax[1].annotate(rf"$\gamma$={gamma}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    plt.tight_layout(); plt.savefig(f"{OUT}/3_gamma_law.png"); plt.close()
    print(f"fig 3 done   slope={slope:.3f}")
    return scale


# --------------------------------------------------------------------------- #
# 4 -- the error is deterministic but not smooth in gamma
# --------------------------------------------------------------------------- #
def fig4(scale):
    gammas = np.arange(200, 501, 5)
    relative = np.array([spiking_error(problem, reference, int(g))[0] * g / scale for g in gammas])

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(gammas, relative, ".-", lw=1.1, ms=5)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label=r"mean of the $1/\gamma$ law")
    ax.fill_between(gammas, 0, relative, alpha=0.12)
    ax.set(xlabel=r"$\gamma$", ylabel=r"error $\times\,\gamma$ / scale",
           title=r"Error is not smooth in $\gamma$ — provision from the formula, don't tune by trial")
    ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/4_fine_structure.png"); plt.close()
    print(f"fig 4 done   spread {relative.min():.2f} to {relative.max():.2f}")


# --------------------------------------------------------------------------- #
# 5 -- remove the coupling and the error disappears
# --------------------------------------------------------------------------- #
def fig5():
    rng2 = np.random.default_rng(1)
    ortho = np.linalg.qr(rng2.standard_normal((48, 48)))[0]      # orthonormal -> weights = 0
    x_ortho = np.zeros(48)
    x_ortho[rng2.choice(48, 5, replace=False)] = rng2.standard_normal(5) * 2
    problem_ortho = no.lasso(ortho, ortho @ x_ortho, lam)
    reference_ortho = no.solve(problem_ortho, backend="reference")

    gammas = np.array([100, 300, 1000, 3000, 10000])
    coupled = np.array([spiking_error(problem, reference, int(g))[0] for g in gammas])
    decoupled = np.array([spiking_error(problem_ortho, reference_ortho, int(g))[0] for g in gammas])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.loglog(gammas, coupled, "o-", lw=1.8, label="overcomplete dictionary (neurons inhibit each other)")
    ax.loglog(gammas, np.maximum(decoupled, 1e-17), "s-", lw=1.8, label="orthonormal dictionary (weights = 0)")
    ax.axhline(2.2e-16, color="k", ls=":", lw=1.2, label="double-precision epsilon")
    ax.set(xlabel=r"$\gamma$ (spikes per unit)", ylabel=r"$\|x - x^*\|$", title="Where the spiking error comes from")
    ax.legend(fontsize=8.5, loc="center left")
    plt.tight_layout(); plt.savefig(f"{OUT}/5_coupling_control.png"); plt.close()
    print(f"fig 5 done   decoupled errors: {decoupled}")


# --------------------------------------------------------------------------- #
# 6 -- sparsity is the point: lambda controls spikes
# --------------------------------------------------------------------------- #
def fig6():
    lams = np.geomspace(lam / 8, lam * 8, 7)
    n_active, spikes = [], []
    for value in lams:
        result = no.solve(no.lasso(dictionary, signal, value), backend="spiking", t_max=120, spikes_per_unit=1000, tol=None)
        n_active.append(result.n_active)
        spikes.append(result.stats["total_spikes"] / 1e6)

    # dense contrast: squared-l2 only (ridge). every neuron fires.
    dense = no.solve(no.elastic_net(dictionary, signal, l1=0.0, l2=0.3), backend="spiking",
                     t_max=120, spikes_per_unit=1000, tol=None)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lams, n_active, "o-", lw=1.8)
    ax[0].axvline(lam, color="C3", ls="--", lw=1.2, label="BIC choice")
    ax[0].set(xscale="log", xlabel="λ", ylabel="active neurons", title="Stronger penalty, fewer active neurons")
    ax[0].legend()

    ax[1].plot(lams, spikes, "o-", lw=1.8, label="LASSO (sparse)")
    ax[1].axhline(dense.stats["total_spikes"] / 1e6, color="C3", ls="--", lw=1.8,
                  label=f"ridge (dense, {dense.n_active} active)")
    ax[1].set(xscale="log", xlabel="λ", ylabel="spikes (millions)", title="Sparse answers mean quiet networks")
    ax[1].legend(fontsize=9)

    plt.tight_layout(); plt.savefig(f"{OUT}/6_sparsity_spikes.png"); plt.close()
    print(f"fig 6 done   active={n_active}")


# --------------------------------------------------------------------------- #
# 7 -- benchmark: accuracy, time, estimated energy per backend
# --------------------------------------------------------------------------- #
def fig7():
    rows = no.benchmark(problem, options={"spiking": {"precision": 1e-3}}, repeats=3)
    no.print_table(rows)
    names = [r["backend"].split("(")[0].split("[")[0] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    ax[0].bar(names, [max(r["error_vs_reference"], 1e-17) for r in rows], color="C0")
    ax[0].set(yscale="log", ylabel=r"$\|x - x^*\|$", title="Accuracy vs exact solver")

    ax[1].bar(names, [r["wall_time_ms"] for r in rows], color="C1")
    ax[1].set(ylabel="ms", title="Wall time (CPU simulation)")

    width = 0.27
    pos = np.arange(len(rows))
    for offset, key, label in [(-width, "energy_cpu_nJ", "CPU"), (0, "energy_gpu_nJ", "GPU"), (width, "energy_loihi_nJ", "Loihi")]:
        values = [r[key] if np.isfinite(r[key]) else 0 for r in rows]
        ax[2].bar(pos + offset, values, width, label=label)
    ax[2].set(yscale="log", xticks=pos, xticklabels=names, ylabel="nJ (estimated)", title="Energy estimate")
    ax[2].legend(fontsize=8.5)
    for a in ax:
        a.tick_params(axis="x", labelrotation=15)

    plt.tight_layout(); plt.savefig(f"{OUT}/7_benchmark.png"); plt.close()
    print("fig 7 done")


if __name__ == "__main__":
    fig0()
    fig1()
    fig2()
    scale = fig3()
    fig4(scale)
    fig5()
    fig6()
    fig7()
    print(f"\nall figures written to {OUT}/")
