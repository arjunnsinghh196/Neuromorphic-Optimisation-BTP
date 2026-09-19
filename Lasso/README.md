# Lasso — Solving LASSO with a Spiking Neural Network

**RnD / BTP Project · Kruthi S (23B1232)**

A library that takes an optimisation problem and solves it by building a network
of artificial neurons, switching it on, and letting it settle. The settled state
is the answer.

```python
import numpy as np, neuroopt as no

prob = no.lasso(Phi, s, lam=0.08)
print(no.describe(prob))                       # which neuron, which weights
res = no.solve(prob, backend="spiking")
print(res.summary())
```

---

## The idea in one paragraph

Normally you solve a maths problem by following steps. This does something else:
it encodes the problem into the wiring and thresholds of a neuron circuit, and
the circuit's equilibrium *is* the solution. Nothing iterates — the network just
relaxes, like water finding its level.

LASSO is the natural first target because its answers are sparse. Sparse answer
means most neurons stay silent, and on event-driven hardware silence is free.

---

## The mapping

Every problem this library handles reduces to one canonical form:

```
minimise   ½ xᵀQx + cᵀx + λ₁·|x| + λ₂·x²
subject to Ax ≤ k,  optionally x ≥ 0
```

and each piece becomes a physical part of the circuit:

| Objective piece | Becomes | Meaning |
|---|---|---|
| `Q` (quadratic) | synaptic weights `W = I − Q`, **zero diagonal** | how neurons inhibit each other |
| `c` (linear) | bias current `−c` | constant input |
| `λ₁` (ℓ1) | **dead zone** | how loud input must be before the neuron reacts |
| `λ₂` (ℓ2 squared) | **firing threshold** `ν_f = 2λ₂ + 1` | charge needed to fire; gain `1/ν_f` |
| `Ax ≤ k` | second neuron layer, feedback via `Aᵀ` | constraint checkers |
| `x ≥ 0` | rectified neurons | one-sided firing |

Changing the regulariser does not mean writing a new solver — it means turning a
knob on the neuron. LASSO and Elastic Net produce a **bit-identical** weight
matrix; only the threshold moves:

```
weights identical           : True
dead zone   lam1            : 0.080 -> 0.080
threshold   nu_f = 2*lam2+1 : 1.000 -> 1.600
```

### The neuron

```
mu_dot = bias − mu − W_syn·sigma − Aᵀv          soma current
vm_dot = shrink(mu, lam1)                        membrane fills (only above the dead zone)
spike when |vm| ≥ nu_f,  then vm −= sign·nu_f    subtractive reset
```

Answer is read off as the firing rate. The constraint layer integrates the
arriving spike train and stays silent whenever its constraint is slack.

---

## Four backends, one dynamical system

| Backend | What it is |
|---|---|
| `spiking` | event-driven integrate-and-fire network (default) |
| `analog` | the same flow in continuous time, integrated numerically |
| `ista` / `fista` | proximal gradient — unit-step forward Euler on the analog flow |
| `reference` | high-accuracy digital solve (SLSQP), for validation only |

They differ in *how* the dynamics are run, not in what is being solved.
`no.compare(prob)` runs all four side by side:

```
backend                      objective        KKT      viol    |x-ref|   nnz
reference(SLSQP)          0.5029807008   9.89e-07  0.00e+00   0.00e+00     6
FISTA                     0.5029807008   1.01e-08  0.00e+00   1.34e-06     6
analog-LCA(euler)         0.5029807008   1.57e-10  0.00e+00   1.32e-06     6
spiking-LCA[filtered]     0.5029807008   1.73e-06  0.00e+00   2.57e-06     6
```

---

## Three errors found in the derivations

Each has a regression test that would catch it coming back.

**1. A neuron was inhibiting itself.** The notes sum over *all* `j` including
`j = i`, so `w_ii = ‖φ_i‖² = 1`. Work through the algebra and the network settles
at `∇f(a) + a + λ·sign(a) = 0` — that extra `+a` makes it an **Elastic Net with
λ₂ = ½, not a LASSO**. The network was silently solving a different problem.
Fix: `w_ii = 0`; a neuron's self-inhibition is its subtractive reset, not a
synapse. The test deliberately puts the diagonal back and confirms the result
matches `elastic_net(λ₂ = 0.5)`.

**2. Constraints could never switch off.** `dy/dt = [Ax − b]₊` is never negative,
so multipliers only ever increase — a constraint violated for one instant can
never go inactive again, making complementary slackness unreachable. Fix: put the
rectifier on the *state*, not the rate of change. `w` accumulates the signed
violation, `v = [w]₊` is the readout. The test overshoots a slack constraint and
asserts the multiplier returns to zero.

**3. `v_t = θ_G(v_{t−1})·(w_t + β(Ax_t − k))` does not parse.** A ReLU output
multiplying a bracket is not a projection and contradicts the prose below it.
Fix: `v_t = [w_t + β(Ax_t − k)]₊`, with an anti-windup floor on `w`.

Also fixed: `‖a‖₂` → `‖a‖₂²` throughout (the unsquared norm has a
block-thresholding prox and a different neuron — the class is named `L2Squared`
so it cannot be written wrong), and the `λ/2` vs `λ` convention.

---

## The measurement: accuracy scales as 1/γ

A spike is a quantum of activation. With `γ` spikes per unit activation per unit
time, smaller quanta mean a more accurate answer at the cost of more spikes.

The obvious prediction is shot noise — random arrivals averaging out like √n,
giving error ∝ 1/√γ. **That is wrong.** Measured over two decades:

| γ | `‖x − x*‖` | spikes |
|---|---|---|
| 100 | 5.0e-04 | 1.2e+05 |
| 1000 | 5.7e-05 | 1.2e+06 |
| 10000 | 4.1e-06 | 1.2e+07 |

Log-log fit over nine points: **exponent −1.010**.

The reason is the **subtractive reset**. After firing, the leftover membrane
charge is carried forward rather than discarded, so it never random-walks — it
stays bounded below one quantum. That makes each neuron a **first-order
sigma-delta modulator**, and sigma-delta error scales with step size, not with
the square root of the event count. A hard reset would throw the remainder away
and restore the √ law.

Two further findings, both measured:

**The error is deterministic.** Repeated runs are bit-identical to sixteen
digits, and the error is quasi-chaotic in γ — 7× worse at γ=300 than at γ=320.
So error is **not monotone in spike budget** and γ cannot be tuned by local
search. Sub-quantum dither collapses the spread from 4.3× to 1.3×.

**It is caused entirely by coupling.** On an orthonormal dictionary (`W = 0`, no
lateral inhibition) the spiking solver matches the continuous flow to machine
precision, identically at every γ:

```
orthonormal dictionary, gamma=   100:  err = 2.428e-13
orthonormal dictionary, gamma=  1000:  err = 2.428e-13
orthonormal dictionary, gamma= 10000:  err = 2.428e-13
analog (no spikes at all)          :  err = 2.449e-13
```

Same neurons, same spikes, same reset — only the coupling removed. A neuron's own
spikes never reach its own soma current (because `w_ii = 0`), so its readout is
clean and **all** the error is injected by its neighbours.

Consequence: a coherent dictionary charges you twice — settling time through
`λ_min`, *and* spike budget through `|W|`. Only the first is in the usual
conditioning discussion.

---

## Install and run

```bash
git clone https://github.com/<your-username>/Neuromorphic-Optimisation-BTP.git
cd Neuromorphic-Optimisation-BTP/Lasso
pip install -e .
pytest tests -q            # 19 tests, ~1.5 s
python examples/demo.py    # nine worked sections
python examples/make_figures.py   # six figures into figures/
```

Dependencies: `numpy`, `scipy` (reference solver only), `matplotlib` (figures
only).

---

## Usage

### Standard problems

```python
no.lasso(Phi, s, lam)                      # ½‖s−Φa‖² + λ‖a‖₁
no.elastic_net(Phi, s, lam1, lam2)         # ... + λ₁‖a‖₁ + λ₂‖a‖₂²
no.ridge(Phi, s, lam)                      # ... + λ‖a‖₂²
no.nnls(Phi, s, lam)                       # lasso with a ≥ 0
no.quadratic_program(Q, c, A, k)           # ½xᵀQx + cᵀx  s.t. Ax ≤ k
no.sparse_qp(Q, c, lam1, A, k)             # ... + λ₁‖x‖₁  s.t. Ax ≤ k
```

### Or build it term by term

```python
prob = (no.LeastSquares(Phi, s) + no.L1(0.05) + no.L2Squared(0.1)).subject_to(
           no.LinearInequality(A, k),
           no.NonNegative(N))
```

### Reading the result

```python
res.x                # answer
res.y                # dual multipliers
res.objective        # objective value
res.violation        # constraint violation (0 = feasible)
res.kkt_residual     # optimality (0 = exact)
res.support          # indices judged non-zero
res.stats            # spikes, gamma, synaptic events
res.summary()        # all of it, printed
```

### If something goes wrong

| Symptom | Fix |
|---|---|
| error too large | `precision=1e-4` (more spikes) |
| `converged=False` *and* answer off | `t_max=3000` |
| too slow | `precision=1e-2` or `spike_resolution=500` |
| constraints violated | increase `t_max` — the dual layer is slower |
| `nnz` differs from reference | `precision=1e-4` |

`converged=False` is often reported while the answer is correct — the stopping
rule is deliberately conservative. Read `kkt_residual` and `violation` instead.

---

## Layout

```
Lasso/
├── neuroopt/
│   ├── objectives.py    term algebra, constraints, canonical form, KKT residuals
│   ├── compile.py       canonical problem -> weights, biases, thresholds
│   ├── solvers.py       the four backends; the spiking loop lives here
│   ├── diagnostics.py   compare(), conditioning_report(), spike_report()
│   └── api.py           solve(), describe(), problem constructors
├── examples/
│   ├── demo.py          nine worked sections
│   └── make_figures.py  six figures
├── tests/               19 tests
└── pyproject.toml
```

The whole spiking neuron is five lines inside `_run_spiking()` in `solvers.py`.
The fourth line — subtract, do not reset to zero — is what sets the accuracy
exponent of the entire solver.

---

## Validated

`pytest tests` — 19 tests:

- all four backends agree on LASSO, Elastic Net, NNLS, constrained QP, and
  ℓ1-regularised constrained QP
- the three correction tests above
- ISTA is unit-step forward Euler on the LCA flow, asserted step by step for 100
  iterations at `atol=1e-12`
- `ν_f = 2λ₂ + 1` reproduces the Elastic Net proximal operator pointwise
- unnormalised dictionaries are rescaled correctly
- the rate readout carries an O(1/T) transient bias; the filtered readout does not
- accuracy improves with spike resolution; higher λ gives fewer spikes
- Ridge is flagged as a poor fit and comes out dense
- non-convex `Q` is rejected rather than silently solved

Randomised cross-check over 12 problems (random M, N, sparsity, λ, family):
spiking agrees with reference to ~1e-6 throughout.

---

## Not implemented

Group sparsity, total variation, nuclear norm — the open question in each case is
which neuron implements the proximal operator, i.e. what replaces `shrink()`.
Group sparsity in particular needs pooled inhibition across a group rather than a
per-neuron dead zone, which is a genuinely different circuit.

Also: equality constraints, on-chip dictionary learning, a Lava backend.

**This is a simulation of a spiking network, not a hardware deployment.** It
models the dynamics, the spike economy and the precision limits of rate coding.
It does not model routing latency, neuron placement or synchronisation barriers.

---

## Next steps

1. **Does `C` scale with dictionary coherence?** Measure `err ≈ C/γ` at coherence
   0.1, 0.5, 0.9 with everything else fixed. If `C` tracks `‖W‖` predictably,
   that gives the spike-cost-of-coherence formula the conditioning discussion is
   missing.
2. **Make dither the default** — reliability beats occasional luck.
3. **New objective classes** (group sparsity first).
4. **Check two "open problems" that may be closed:** the LCA convergence rate is
   known (Balavoine–Romberg–Rozell: exponential once the active set settles,
   governed by the eigenvalue restricted to the support), and the
   Euler–Lagrange question has a route via the Bregman Lagrangian plus
   Moreau–Yosida smoothing for the non-smooth case.
