# Phase 8 - UCI Iris real-data pilot

Date: 2026-11-13

Status: completed with one retained initial failure and a proposal-aligned
certificate refinement; final refined result `PHASE8_IRIS_PILOT=PASS`.

## Research question

Does the lower-variance, exact-feasible SAGE gradient from Phase 7 translate
into successful end-to-end learning on a small real dataset without exceeding a
reasonable matched black-box compute budget?

This is a pilot and not a dataset-superiority claim. Its purpose is to validate
the training protocol and detect failure modes before using a larger task.

## Data and task

The pilot uses the UCI Iris dataset (150 plants, three balanced species, four
continuous measurements), DOI `10.24432/C56C76`, under CC BY 4.0. The official
`bezdekIris.data` file is committed with SHA-256 and attribution in
`data/phase8/README.md`.

- Inputs: sepal length and sepal width.
- Regression targets: petal length and petal width.
- Disjunction: three axis-aligned petal-space boxes, one per species.
- Split per seed: 30 train, 10 validation and 10 test plants per species.
- Input/output normalization and every box bound are fit from the training
  split only.
- Species labels are used to fit training boxes and report branch accuracy;
  validation/test targets never determine constraints or normalization.

The output task retains continuous within-branch derivatives while requiring
discontinuous switching among species regions. This directly exercises the
local implicit term and switching residual instead of reducing the layer to a
pure one-hot argmax.

## Proposal contract

- exact branch-wise Euclidean forward projection, equations (7)-(10);
- certified stability/competitor logic, equations (11)-(12);
- compact `q(u) proportional to (1 - norm(u)^2)^2`, `beta=2`;
- fixed `rho=0.5` for the full pilot;
- fixed reference branch within each SAGE sample;
- local implicit plus residual score correction, equations (17)-(18);
- no averaging between feasible branches;
- common antithetic perturbations and matched `q`, `rho`, `N` for black-box
  and SAGE;
- exact final forward evaluation and original-logic feasibility checks;
- branch/relaxation/global-search counts and complete training time reported.

## Frozen training design

```text
seeds:                 1, 2, 3, 4, 5
network:               2 -> 8 tanh -> 2
optimizer:             Adam, learning rate 0.02
updates:               60
batch size:            12
black-box/SAGE samples: 8 antithetic compact-q samples per observation
radius:                0.5
methods:               hard-local, STE, black-box, SAGE
```

Every method receives the same seed-specific initialization, split, minibatch
sequence and update count. Black-box and SAGE additionally receive identical
per-observation perturbations. Method execution order rotates across seeds to
reduce timing-order bias. Compilation warm-up is excluded from recorded time.

## Acceptance criteria fixed before the five-seed run

1. Analytic box projection and certified backend agree below `1e-6` in output
   and objective.
2. Every train/validation/test output is logically feasible, with no unresolved
   training samples.
3. Mean SAGE test MSE is at most 80% of its matched initial-network MSE.
4. Mean SAGE test MSE is no more than `1.10x` black-box and mean branch accuracy
   is no more than 0.10 below black-box.
5. Mean SAGE test MSE is no greater than hard-local test MSE.
6. SAGE performs at least 25% fewer perturbed global searches than black-box.
7. Mean SAGE end-to-end training time is no more than `1.50x` black-box.

All seven criteria must pass. Paired mean differences and conservative 95%
Student-t intervals across seeds are reported but are not used to relax the
gate. STE remains a required low-cost heuristic comparator.

## Preliminary implementation checks

- Dataset/split/train-only geometry tests: `196 / 196` passed.
- Analytic/certified backend parity tests: `100 / 100` passed.
- Matched initialization and one-step training tests: `13 / 13` passed.
- A two-update plumbing run completed every method with exact feasibility. It
  was not used to choose or change the frozen 60-update acceptance thresholds.

## Reproduction

```text
julia --project=. scripts/benchmark_phase8_iris.jl
```

The runner writes per-seed results, trajectories and a machine-readable summary
to `phase8_iris_results/`. Numerical results and limitations will be appended
after the frozen run.

## Frozen acceptance run 1 - retained failure

The first five-seed run kept the exact preregistered training design and
produced:

| Method | Mean test MSE | Branch accuracy | Mean training seconds | Perturbed global searches |
|---|---:|---:|---:|---:|
| hard-local | 0.36483 | 0.6600 | 1.9716 | 0 |
| STE | 0.28174 | 0.7400 | 1.2303 | 0 |
| black-box | 0.29117 | 0.7267 | 11.0579 | 28,800 |
| SAGE | **0.27459** | 0.7200 | 11.3448 | 22,704 |

Correctness, learning, matched quality, hard-local improvement and wall-time
gates passed. Global-search reduction was only `21.17%`, below the frozen `25%`
criterion. Therefore the overall result was retained as:

```text
PHASE8_IRIS_PILOT=FAIL
```

This failure exposed an implementation gap. When the radius-wide switching
certificate was inconclusive, every sample immediately entered a global search,
even when the already-computed fixed-reference projection was provably better
than every competitor at that individual perturbed point.

## Safe pointwise certificate refinement

The estimator now performs a second, sample-specific proof before any global
search. For fixed-reference projection objective `U_ref(w)` and every remaining
competitor group `G`, it requires the strict certified inequality

```text
geometric_lower_bound(G, w) > U_ref(w) + tolerance.
```

If all groups satisfy it, the reference branch is the unique exact winner at
that sample. The equation (18) residual is exactly zero, so the global search
can be skipped without altering the estimator. Strict inequality preserves the
existing deterministic tie rule. Failure to prove dominance falls back to the
same restricted or lazy exact search as before.

This is an implementation of the proposal's certificate/pruning architecture,
not a new heuristic, threshold change, radius change or training-budget change.
`SAGEEstimate` now reports `sample_stability_certificates` separately.

## Refined acceptance run

The five seeds, splits, initialization, minibatches, perturbations, network,
optimizer, `rho`, `N`, updates and all seven original gates were unchanged.

```text
PHASE8_IRIS_PILOT=PASS
```

| Method | Mean test MSE | Branch accuracy | Mean training seconds | Perturbed global searches | Pointwise certificates |
|---|---:|---:|---:|---:|---:|
| hard-local | 0.36483 | 0.6600 | 1.9365 | 0 | 0 |
| STE | 0.28174 | 0.7400 | 1.2440 | 0 | 0 |
| black-box | 0.29117 | 0.7267 | 11.0772 | 28,800 | 0 |
| SAGE | **0.27459** | 0.7200 | **7.3869** | **3,083** | 19,621 |

Final structural results:

```text
maximum projection output parity error:    7.0517e-8
maximum projection objective parity error: 2.4863e-15
logical feasibility rate:                  100%
unresolved SAGE samples:                   0
global-search reduction:                   89.30%
SAGE/black-box training-time ratio:         0.6669
```

The pointwise proof removed additional searches but did not change final SAGE
quality: its seeded test metrics are numerically identical to the initial run.

Paired five-seed comparisons:

```text
SAGE - black-box test MSE: -0.01658, 95% CI [-0.05825, 0.02509]
SAGE - hard-local test MSE: -0.09024, 95% CI [-0.22700, 0.04652]
SAGE - black-box accuracy:  -0.00667, 95% CI [-0.02517, 0.01184]
```

## Interpretation and limitations

SAGE achieved the lowest mean test MSE while using approximately 89% fewer
perturbed global searches and 33% less measured training time than matched
black-box smoothing. Exact feasibility and backend parity were preserved.

However, every paired 95% interval above includes zero. Iris has only 150
observations and five seeds are insufficient for a statistically conclusive
quality-superiority claim. STE also remains extremely competitive: its MSE is
close to SAGE, its mean branch accuracy is highest, and it is much faster.

The paper-safe conclusion is therefore: the first real-data pilot validates the
proposal architecture and produces a promising efficiency/quality signal, but
does not establish broad or statistically significant superiority. A larger
Phase 9 task must test whether the signal survives more observations, more
branches, stronger backbones and additional seeds.

## Regression and backward-compatibility verification

The first full-suite invocation exposed two missing imports (`mean` and `std`)
inside the new test file. The algorithmic tests had passed when run alone
because the invoking command imported `Statistics`. The test was made
self-contained with explicit imports; no algorithm or acceptance result changed.

Final package regression:

```text
Test Summary:     | Pass  Total   Time
DisjunctiveNet.jl | 3064   3064  55.3s
```

Earlier proposal gates were rerun after the pointwise certificate change:

```text
Phase 4 gradient verification: PASS
  analytic gradient error unchanged: 0.0187973372
  switching global searches:         9 / 1000

Phase 6 matched-budget gate: PASS
  SAGE global searches:              45 / 192 black-box
  SAGE/black-box wall-time ratio:     0.7356

Phase 7 completed scaling gate: PASS
  SAGE global searches:              23 / 96 black-box
  SAGE/black-box wall-time ratio:     0.8825
  variance and RMSE values:           unchanged
```

The unchanged gradient-quality values alongside fewer solver calls verify that
the new pointwise proof removes only samples with an exactly zero switching
residual; it does not alter equations (17)-(18).
