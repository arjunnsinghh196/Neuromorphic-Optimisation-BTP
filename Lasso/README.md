# neuroopt — LASSO on Spiking Neural Networks

Solve LASSO (and related convex problems) by running a network of spiking neurons. The objective is turned into neuron parameters directly — no training, no approximation — and the network's steady-state firing rates are the solution.

```
min  ½ ‖signal − dictionary·x‖²  +  λ ‖x‖₁
```

| Objective piece | Becomes |
|---|---|
| dictionaryᵀ·dictionary (the Gram matrix) | recurrent weights **I − Gram** |
| −dictionaryᵀ·signal | constant input current |
| λ | firing **dead zone**: a neuron stays silent until its input exceeds λ |
| squared-L2 weight (elastic net) | firing **threshold** 2·l2 + 1 |
| A·x ≤ b | a second layer of **constraint neurons** |
| x ≥ 0 | **rectified** neurons |

Each atom in the dictionary gets one neuron. A neuron is driven by how well its atom matches the signal and inhibited by its neighbours in proportion to how similar their atoms are. The best-matching neurons cross the dead zone, fire, and suppress the rest. Silence is zero; sparsity in the answer is literally silence in the network.

---

## Contents

1. [Install and run](#1-install-and-run)
2. [API](#2-api)
3. [Design](#3-design)
4. [Choosing λ](#4-choosing-λ)
5. [Reading a result](#5-reading-a-result)
6. [Benchmarking](#6-benchmarking)
7. [Testing](#7-testing)
8. [Theory, briefly](#8-theory-briefly)
9. [References](#9-references)

---

## 1. Install and run

Needs `numpy` and `scipy` (scipy only for the reference solver).

```bash
.\venv-lasso\Scripts\Activate.ps1          # Windows
pip install numpy scipy pytest

python examples/lasso_demo.py              # tune λ, solve, benchmark four sizes
python -m pytest tests                     # 22 tests
```

```
Lasso/
├── neuroopt/
│   ├── objectives.py   problem terms and the canonical form
│   ├── compile.py      canonical form → network
│   ├── solvers.py      reference / ISTA / analog / spiking
│   ├── api.py          solve(), describe(), lasso(), ...
│   ├── tuning.py       find_lambda()
│   └── benchmark.py    benchmark(), print_table()
├── examples/lasso_demo.py
└── tests/test_lasso.py
```

---

## 2. API

### Build a problem

```python
import neuroopt as no

problem = no.lasso(dictionary, signal, lam)            # ½‖s − Dx‖² + λ‖x‖₁
problem = no.lasso(dictionary, signal, lam, nonneg=True)
problem = no.nnls(dictionary, signal)                  # same as above, λ = 0
problem = no.elastic_net(dictionary, signal, l1, l2)   # adds l2‖x‖₂²
problem = no.sparse_qp(gram, linear, l1, matrix=A, rhs=b)
problem = no.quadratic_program(gram, linear, matrix=A, rhs=b, low=0, high=1)
```

Or compose terms directly:

```python
problem = no.Problem([no.LeastSquares(D, s), no.L1(0.1, size=D.shape[1])])
problem = problem.subject_to(no.NonNegative())
```

### Choose λ

```python
lam, result, sweep = no.find_lambda(dictionary, signal, criterion="bic")
```

`criterion` is `"bic"` (fewer atoms) or `"aic"` (better fit). `sweep` is a list of dicts — one per λ tried — with `lambda`, `score`, `rss`, `n_active`, `objective`, for plotting.

### Solve

```python
result = no.solve(problem, backend="spiking")
result = no.solve(problem, backend="spiking", precision=1e-3, t_max=2000)
result = no.solve(problem, backend="reference")   # exact, for checking
```

Backends and their options:

| backend | what it is | useful options |
|---|---|---|
| `reference` | SciPy SLSQP; slow, exact | `tol` |
| `ista`, `fista` | proximal gradient; the standard baseline | `step`, `max_iter`, `tol`, `x0` |
| `analog` | the network ODE, Euler-integrated | `dt`, `t_max`, `tol`, `x0` |
| `spiking` | integrate-and-fire network | `precision`, `spikes_per_unit`, `t_max`, `readout`, `x0`, `active_threshold` |

All accept `active_threshold` (default 1e-9; `"auto"` for spiking = 3× its noise floor). `x0` warm-starts.

### Inspect

```python
print(no.describe(problem))     # neuron model, weights, thresholds, condition number
net = no.network_for(problem)   # the compiled Network object: .weights, .dead_zone, .threshold, ...
```

### Benchmark

```python
rows = no.benchmark(problem, repeats=3)
no.print_table(rows)
```

### The `Result` object

| field | meaning |
|---|---|
| `x` | solution, with values below `active_threshold` set to exactly 0 |
| `x_raw` | solution before truncation |
| `active` / `n_active` / `sparsity` | indices of non-zeros, how many, fraction of zeros |
| `active_threshold` | what counted as zero |
| `objective` | value of the objective at `x` |
| `kkt_residual` | distance from optimality (0 = exact) |
| `violation` | largest constraint violation (0 = feasible) |
| `converged`, `iterations`, `wall_time_s` | |
| `multipliers` | Lagrange multipliers if constrained |
| `history` | time series of objective / KKT for plotting |
| `stats` | backend-specific numbers (spikes, flops, dt, …) |
| `summary()` | printable report |

---

## 3. Design

### One canonical form

Every problem type reduces to

```
min  ½ xᵀ·gram·x + linearᵀx + l1_weightᵀ|x| + l2_weightᵀx²
s.t. constraint_matrix·x ≤ constraint_rhs,  x ≥ 0 (optional)
```

and every solver consumes exactly that (`CanonicalProblem`). This keeps the code small: a new problem type is a new `Term` class, a new solver handles one input type. Nothing outside this form is accepted — a non-convex Gram, an unsquared L2 norm, an equality constraint all raise at build time rather than produce a wrong number.

### Compile is separate from run

`compile_network()` turns the canonical problem into a `Network` (weights, input current, dead zone, threshold). Every backend runs the *same* compiled network. The spiking solver is not a separate implementation of LASSO; it's the same object executed with spikes instead of real numbers. You can inspect the network without solving (`network_for`).

### No self-synapse

Weights are `I − gram`, whose diagonal is `1 − gram[i,i]`. In a spiking neuron, a cell's own inhibition is its **reset**, not a synapse. If the diagonal were left in, the neuron would inhibit itself twice and the problem actually solved would be an elastic net with l2 = ½. So the compiler zeroes the diagonal — which only works if `gram[i,i] = 1`, i.e. unit-norm atoms. If they aren't, the compiler rescales the variables (`rescale_to_unit_diagonal`), solves, and maps back. `describe()` tells you when this happened.

### Truncation and the active threshold

A rate-coded output is never exactly zero — a neuron that fires once in a while still has a tiny rate. So `Result.x` sets everything below `active_threshold` to exactly 0, and `x_raw` keeps the untouched values. The exact solvers use 1e-9. The spiking solver estimates its own noise floor (`(|weights|·|activation|) / spikes_per_unit`) and uses 3× that. This makes "active" mean the same thing across backends, and makes sparsity an honest number.

### Spike resolution is chosen for you

`spikes_per_unit` (γ) is how many spikes represent one unit of activation. More spikes → finer quantisation → more accurate, more expensive. `"auto"` runs a 60-time-unit analog pre-pass, estimates the noise each neuron will receive from its neighbours, and picks the smallest γ that meets `precision` (default 1%, relative to the largest coefficient). Constraint feedback is included in that estimate.

### Convergence for spiking

The output can't settle tighter than its own noise floor, so the convergence check uses `max(tol, noise_floor)` and requires two consecutive settled checks. Before this, spiking runs always used the full `t_max` because `tol=1e-6` was below the jitter.

### Warm start

`x0` seeds every iterative backend. For the network backends it sets the membrane so the initial activation equals `x0`. `find_lambda` uses this to run its sweep in a fraction of the cold-start time.

---

## 4. Choosing λ

λ is a hyperparameter. The library doesn't assume you know it.

`find_lambda` sweeps 30 values geometrically from `lambda_max` (the smallest λ for which the solution is all zeros — it's `‖dictionaryᵀ·signal‖∞`) down to `lambda_max × 1e-3`, warm-starting each solve from the previous one. For each λ it:

1. solves the LASSO (FISTA by default — fast; pass `backend="spiking"` to sweep on the network)
2. takes the active set and **refits it by least squares** (relaxed LASSO). This removes the shrinkage that LASSO applies to large coefficients, so the score judges *which atoms* were picked, not how much they were shrunk
3. scores with BIC or AIC:

```
variance = RSS_refit / (n_samples − n_active)
BIC      = n_samples · log(variance) + n_active · log(n_samples)
AIC      = n_samples · log(variance) + 2 · n_active
```

Lower is better. The first term rewards fit; the second punishes extra atoms. The degrees-of-freedom correction `n_samples − n_active` is what stops the score from going to −∞ when a square or overcomplete dictionary interpolates the signal exactly — without it, the sweep always chose the smallest λ.

Things to know:

- **BIC is stricter than AIC.** BIC's penalty grows with `log(n_samples)`; AIC's is constant 2. BIC picks fewer atoms. For sparse recovery, use BIC.
- **With noise you will sometimes get one or two small extra atoms.** That's a property of LASSO, not a bug; the planted atoms are always found if their magnitude is above the noise.
- **You need more samples than atoms for the criterion to mean anything.** With `n_samples ≤ n_atoms` the dictionary can fit anything and `dof` hits zero — those λ values are scored `inf` and skipped.
- **A pure-noise signal correctly returns an empty support** at a large λ. The tuner isn't forced to pick something.

---

## 5. Reading a result

Example from the demo (10 samples, 5 atoms, 2 planted, noise 0.05, BIC chose λ = 0.448):

```
  neuron    x_spiking        x_ref    x_refit     x_true  status
       0    +0.000000    +0.000000     +0.000     +0.000  -
       1    +0.127579    +0.127567     +0.663     +0.631  ACTIVE
       2    +0.000000    +0.000000     +0.000     +0.000  -
       3    -0.141323    -0.141333     -0.677     -0.677  ACTIVE
       4    +0.000000    +0.000000     +0.000     +0.000  -

  active threshold    : 4.24e-04
  true support        : [1, 3]
  reference support   : [1, 3]
  spiking support     : [1, 3]
  |x_spiking - x_ref| : 1.53e-05
```

- `x_spiking` vs `x_ref` — the spiking network vs the exact solver. They agree to 1e-5 at the requested 1e-3 precision. **This is the correctness claim.**
- `x_true` — what was planted. The LASSO coefficients (`x_ref`) are much smaller: LASSO shrinks every non-zero toward zero by roughly λ, and BIC chose a large λ because it scores the *support*, not the magnitudes. Both solvers give the right answer to the LASSO problem; the LASSO problem is a biased estimate of `x_true`.
- `x_refit` — least squares on the chosen support, returned in the sweep table. This undoes the shrinkage and is within noise of `x_true`. Use the spiking network to find *which* atoms, and `x_refit` for *how much*.
- `active threshold` — anything below this was set to zero.
- `status` — ACTIVE means the neuron fired regularly; `-` means it stayed inside its dead zone.

From `summary()`:

| line | meaning |
|---|---|
| `KKT residual` | 1e-16 means exact. ~1e-5 for spiking is the spike-noise floor, not an error |
| `converged` | did two consecutive checks show the output settled within tolerance / noise floor |
| `spikes_per_unit` | the γ that was chosen |
| `total_spikes` | the cost on event-driven hardware |
| `synaptic_events` | spikes × fan-out; every spike is delivered to every neighbour with a non-zero synapse |
| `flops` | cost of simulating this on a CPU |

---

## 6. Benchmarking

`benchmark(problem)` runs every backend on the same problem and reports, per backend:

**Accuracy** — `error_vs_reference` (‖x − x_ref‖), `support_matches`, `objective_gap`, `kkt_residual`.

**Convergence** — `converged`, `iterations`, `sim_time` (network time units), `iterations_to_kkt_1e-4` (how far into the run the KKT residual first dropped below 1e-4), `wall_time_ms` (median of `repeats` runs).

**Work** — `flops` (counted, not measured: one dense matrix-vector product per iteration = 2n² flops, plus O(n) for the rest), `total_spikes`, `synaptic_events`, `neuron_updates`.

**Energy estimate** — `energy_cpu_nJ`, `energy_gpu_nJ`, `energy_loihi_nJ`.

### How energy is estimated

Energy is *estimated* from operation counts and published per-operation costs. It is not measured. The numbers are in `neuroopt.ENERGY_PJ` and can be overridden with `benchmark(..., energy_pj={...})`.

| platform | cost | where it comes from |
|---|---|---|
| CPU | 200 pJ / flop | desktop i7: ~95 W TDP ÷ ~450 GFLOPS peak FP32. Real small-matrix code is memory-bound and does worse |
| GPU | 20 pJ / flop | NVIDIA A100: 400 W ÷ 19.5 TFLOPS FP32. Only reachable at full occupancy; a 10×10 problem would never get close |
| Loihi | 23.6 pJ / synaptic event, 81 pJ / neuron update | Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro 2018, Table 2 |

`energy_cpu` and `energy_gpu` = flops × cost. `energy_loihi` = synaptic_events × 23.6 pJ + neuron_updates × 81 pJ.

The CPU/GPU figures are for the *dense simulation* of the network. The Loihi figure counts only the spikes that actually happen. That asymmetry is the whole argument for event-driven hardware: on a CPU, a silent neuron still costs its row of the matrix-vector product every iteration; on Loihi it costs nothing but its own update.

### Sample output (demo, seed 42)

```
20x10, 3-sparse, λ = 0.064 (BIC)
backend                  |x-ref| support       KKT  conv   iters       ms     flops    spikes   CPU nJ   GPU nJ Loihi nJ
reference(SLSQP)        0.00e+00    True  8.96e-07  True      12      1.0         -         -        -        -        -
FISTA                   1.08e-06    True  8.30e-10  True     131      2.2  3.67e+04         - 7.34e+03      734        -
analog-LCA              1.08e-06    True  1.38e-10  True      83      1.8  2.49e+04         - 4.98e+03      498        -
spiking-LCA[filtered]   5.83e-05    True  7.38e-05  True     600     10.2  1.92e+05   2.7e+05 3.84e+04 3.84e+03 5.79e+04
```

How to read it honestly:

- **All four agree on support and on x to within the requested precision.** That's the correctness claim.
- **The spiking solver is the slowest on a CPU** (10 ms vs 2 ms). Of course it is — simulating spikes on a CPU is the worst of both worlds. The CPU column is not the argument.
- **At these sizes the Loihi estimate is in the same range as a GPU.** 58 µJ vs 4 µJ for GPU, 38 µJ for CPU. Tiny problems don't show the advantage because `neuron_updates` (every neuron, every tick) dominates, and that term scales with n·ticks regardless of sparsity.
- **The advantage appears when the problem is large and sparse.** Synaptic events scale with (active neurons × fan-out × spikes); flops scale with n² × iterations. At n = 1000 with 1% active, the dense matvec is 2·10⁶ flops per iteration while the spiking network delivers ~10⁴ events per tick. Intel's own measurement of LCA on Loihi vs FISTA on a Core i7 (Davies et al. 2021, *Advancing Neuromorphic Computing with Loihi*) shows the gap opening with problem size — roughly 50× lower energy at a few hundred atoms, growing from there.

To run a larger benchmark:

```python
rng = np.random.default_rng(0)
D = rng.standard_normal((200, 400)); D /= np.linalg.norm(D, axis=0)
x = np.zeros(400); x[rng.choice(400, 10, replace=False)] = rng.uniform(0.5, 2, 10)
s = D @ x + 0.02 * rng.standard_normal(200)
lam, *_ = no.find_lambda(D, s)
no.print_table(no.benchmark(no.lasso(D, s, lam), options={"spiking": {"precision": 1e-2}}))
```

---

## 7. Testing

`tests/test_lasso.py` — 22 tests, ~1 second. They split into:

**Normal path**
- all four backends agree with the reference on x (to 1e-5 / 5e-3) and on support
- the reference solution satisfies KKT to 1e-6
- truncation leaves no values in (0, threshold); `x_raw` is preserved
- the spiking solver is deterministic: two runs give identical spike counts and output

**Edge cases**
- zero signal → zero solution, zero spikes
- λ ≥ λ_max → all zeros on every backend
- λ = 0 → ordinary least squares (checked against `np.linalg.lstsq`)
- orthonormal dictionary → spiking is exact to 1e-8 (no coupling, no spike noise — the key claim in §8)
- non-unit-norm dictionary → `rescaled` flag set, answer still matches reference
- a single atom (n = 1)
- overcomplete dictionary (10 samples, 40 atoms)
- non-negative LASSO → no negative entries, zero violation
- constrained QP with `Ax ≤ b` → all backends agree, constraint satisfied to 5e-3 on spiking
- warm start converges in fewer iterations than cold start

**λ tuning**
- noisy 60×20 problem: BIC finds all planted atoms (magnitude ≥ 0.5), no more than 7 total
- an invalid criterion raises

**Error paths** — each raises `ValueError`: shape mismatch, negative λ, non-PSD Gram, ISTA given constraints, unknown backend name.

Things the tests deliberately don't claim: that the LASSO support equals the planted support (LASSO is biased; with noise it picks extras), or that spiking converges to 1e-6 (it converges to its noise floor).

---

## 8. Theory, briefly

**The network** (continuous time):

```
membrane'  = −membrane + weights·activation + input_current − constraint_matrixᵀ·multipliers
activation = shrink(membrane, dead_zone) / threshold
```

`shrink(u, λ) = sign(u)·max(|u| − λ, 0)` — soft thresholding.

**Why the fixed point is the optimum.** At rest `membrane' = 0`, so `membrane − activation = −(gram·activation + linear + constraint_matrixᵀ·multipliers)` — minus the gradient of the smooth part. And `membrane − activation` lies in `dead_zone·∂|activation| + 2·l2·activation` by the definition of shrink. Put together: `0 ∈ ∇smooth + λ·∂‖x‖₁ + Aᵀv`. That is the KKT condition. `kkt_residual` measures the distance from it.

**The spiking neuron** replaces `activation` with spikes:

```
soma_current ← soma_current + dt·(input − soma_current) − synapse_weights·spikes
membrane     ← membrane + dt·shrink(soma_current, dead_zone)
fire when |membrane| ≥ threshold/γ;  membrane −= sign·threshold/γ      (subtractive reset)
```

**Why the error is 1/γ, not 1/√γ.** Shot noise would give √. But the reset is subtractive — leftover charge carries forward — so each neuron is a sigma-delta modulator and its quantisation error is bounded by one quantum. The error neuron *i* sees is ≈ `(|weights|·|activation|)ᵢ / γ`. Linear. Halving the error costs 2× spikes. It's also deterministic (two runs are identical) and comes entirely from neighbours: with `weights = 0` the spiking solver matches the analog one to machine precision — which is exactly what the orthonormal-dictionary test checks.

**ISTA with step 1 is forward Euler on this ODE.** The usual 1/λ_max step limit is a discretisation artefact. In continuous time there is no step; convergence time is ~1/λ_min, independent of λ_max.

---

## 9. References

- Rozell, Johnson, Baraniuk, Olshausen. *Sparse coding via thresholding and local competition in neural circuits.* Neural Computation, 2008. — the LCA
- Shapero, Zhu, Hasler, Rozell. *Optimal sparse approximation with integrate and fire neurons.* Int. J. Neural Systems, 2014. — spiking LCA
- Beck, Teboulle. *A fast iterative shrinkage-thresholding algorithm.* SIAM J. Imaging Sci., 2009. — FISTA
- Davies et al. *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning.* IEEE Micro, 2018. — the 23.6 pJ / 81 pJ figures
- Davies et al. *Advancing Neuromorphic Computing with Loihi: A Survey of Results and Outlook.* Proc. IEEE, 2021. — LCA on Loihi vs FISTA on CPU, measured
- Meinshausen. *Relaxed Lasso.* Comput. Stat. Data Anal., 2007. — refit-then-score, used in `find_lambda`
