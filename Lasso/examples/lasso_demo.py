"""
LASSO end to end: tune lambda, solve on the spiking network, benchmark all backends.

Edit SETTINGS and CASES at the top. Everything below just runs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so it runs without installing
import neuroopt as no  # noqa: E402

SEED = 42            # None for a new random problem each run
NOISE = 0.05         # noise std added to the signal (0 = clean)
CRITERION = "bic"    # "bic" or "aic" for lambda selection
PRECISION = 1e-3     # spiking readout precision target
T_MAX = 4000         # spiking simulation time

CASES = [            # (samples, atoms, non-zeros); more samples than atoms so lambda is identifiable
    (4, 2, 1),
    (6, 3, 2),
    (10, 5, 2),
    (20, 10, 3),
]


def make_problem(n_samples, n_atoms, n_nonzero, rng, noise=0.0):
    dictionary = rng.standard_normal((n_samples, n_atoms))
    dictionary /= np.linalg.norm(dictionary, axis=0)
    x_true = np.zeros(n_atoms)
    where = rng.choice(n_atoms, n_nonzero, replace=False)
    x_true[where] = rng.choice([-1, 1], n_nonzero) * rng.uniform(0.5, 2.0, n_nonzero)
    signal = dictionary @ x_true + noise * rng.standard_normal(n_samples)
    return dictionary, signal, x_true


def run_case(dictionary, signal, x_true, label):
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print("dictionary (unit-norm columns):")
    print(np.round(dictionary, 3))
    print("signal :", np.round(signal, 3))
    print("x_true :", np.round(x_true, 3))

    # 1. choose lambda
    lam, _, sweep = no.find_lambda(dictionary, signal, criterion=CRITERION)
    x_refit = next(row["x_refit"] for row in sweep if row["lambda"] == lam)
    print(f"\nlambda sweep ({CRITERION.upper()}, {len(sweep)} values from "
          f"{sweep[0]['lambda']:.3g} down to {sweep[-1]['lambda']:.3g})")
    print(f"  chosen lambda = {lam:.4g}")

    # 2. solve
    problem = no.lasso(dictionary, signal, lam)
    reference = no.solve(problem, backend="reference")
    spiking = no.solve(problem, backend="spiking", precision=PRECISION, t_max=T_MAX)

    print(f"\n  {'neuron':>6}  {'x_spiking':>11}  {'x_ref':>11}  {'x_refit':>9}  {'x_true':>9}  status")
    for i in range(len(spiking.x)):
        status = "ACTIVE" if spiking.x[i] != 0 else "-"
        print(f"  {i:>6}  {spiking.x[i]:>+11.6f}  {reference.x[i]:>+11.6f}  "
              f"{x_refit[i]:>+9.3f}  {x_true[i]:>+9.3f}  {status}")

    true_active = [int(i) for i in np.flatnonzero(x_true)]
    print(f"\n  active threshold    : {spiking.active_threshold:.2e}")
    print(f"  true support        : {true_active}")
    print(f"  reference support   : {[int(i) for i in reference.active]}")
    print(f"  spiking support     : {[int(i) for i in spiking.active]}")
    print(f"  |x_spiking - x_ref| : {np.linalg.norm(spiking.x - reference.x):.2e}")
    print(f"  spikes per unit     : {spiking.stats['spikes_per_unit']:.0f}")
    print(f"  total spikes        : {spiking.stats['total_spikes']:.3g}")

    # 3. benchmark every backend
    print("\nbenchmark:")
    no.print_table(no.benchmark(problem, options={"spiking": {"precision": PRECISION, "t_max": T_MAX}}, repeats=1))
    print()
    return spiking, reference


if __name__ == "__main__":
    seed = SEED if SEED is not None else int(np.random.default_rng().integers(0, 10_000))
    rng = np.random.default_rng(seed)
    print(f"seed = {seed}\n")

    for n_samples, n_atoms, n_nonzero in CASES:
        dictionary, signal, x_true = make_problem(n_samples, n_atoms, n_nonzero, rng, NOISE)
        run_case(dictionary, signal, x_true, f"{n_samples}x{n_atoms}, {n_nonzero}-sparse")
