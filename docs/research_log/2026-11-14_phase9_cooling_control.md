# Phase 9 - Native-disjunction cooling-control application

Date: 2026-11-14

Status: pilot acceptance thresholds frozen after the recorded smoke run; the
multi-seed pilot has not yet been observed at the time of this entry.

## Research question

On a cooling-control task whose constraints are natively input-dependent and
disjunctive, can certified and geometry-aware SAGE retain exact feasibility,
match black-box learning quality and reduce complete solver work under matched
perturbation budgets?

## Source recovery before implementation

The official ICML 2026 paper specifies:

- input `x = (Ta, H, w, price, dr, mf, mc)`;
- output `y = (fan, chiller, pump) in [0,1]^3`;
- a cooling-load inequality, always-on pump bound and seven
  input-activated disjunctive rules;
- a `7 -> 8 -> 8 -> 3` ReLU MLP;
- IID and OOD distributions, test seeds and 2,000 points per test split;
- base pretraining for five epochs and projection fine-tuning for one epoch;
- MSE and original-rule constraint satisfaction (CSAT).

The public package repository and its complete 21-commit history contain the
library, examples and tests, but no cooling-control experiment code or data.
The paper explicitly says that the exact numeric configuration is in the
accompanying code, yet that code is not present in the linked public package.
Web/code searches for the unique rule and parameter names did not find another
official artifact.

Consequently, Phase 9 is an **honest paper-spec reconstruction**, not an exact
reproduction of unpublished numeric parameters. Every reconstructed constant
must live in one typed configuration and be printed/saved with results. The
paper-specified distributions, architecture and logical rule consequents are
kept verbatim; missing activation thresholds, load coefficients, oracle
weights and noise level are declared reconstruction choices.

Primary sources:

- https://proceedings.mlr.press/v306/pal26a.html
- https://github.com/li-group/DisjunctiveNet.jl

## Paper-specified rules retained

For active rules:

```text
R1 hot:        chiller >= 0.45 OR fan >= 0.75
R2 cold:       chiller <= 0.05 OR fan >= 0.60
R3 humid:      fan <= 0.70 OR chiller >= 0.30
R4 high load:  fan >= 0.65 OR chiller >= 0.35
R5 demand:     power(y) <= Pdr OR (fan <= 0.50 AND chiller <= 0.40)
R6 fan maint:  fan <= 0.40 OR chiller >= 0.40
R7 chill maint:chiller <= 0.40 OR fan >= 0.70
```

## Hard requirements fixed before smoke tests

1. Dataset generation, oracle targets and splits are deterministic from stored
   seeds; normalization is fit on training data only.
2. Oracle labels come from exact branch-wise QP projection and every oracle
   output passes the original-rule validator.
3. All constrained methods use the same pretrained MLP initialization and the
   same per-seed training order.
4. Black-box and certificate-only SAGE use matched compact-`q`, `rho`, `N` and
   common antithetic samples. Geometry SAGE uses the same `rho` and `N` with
   exact `q/p_z` correction.
5. Every constrained final forward uses exact lazy branch search. Projected
   outputs are never averaged.
6. Report IID and OOD MSE/CSAT, mode agreement, trajectories, global/local
   solve counts, certificate counts, geometry overhead, ESS, wall time and
   paired seed uncertainty.
7. Zero unsafe or logically infeasible accepted outputs. All unresolved cases
   are counted and fail the safety gate.
8. Smoke runs choose feasible compute budgets only. Quality/speed thresholds
   are frozen before the final multi-seed run and failures are retained.

## Implementation entries

### 2026-11-14 - native problem, exact oracle and training harness

Added `scripts/phase9_cooling_common.jl` with one typed `CoolingConfig`, the
paper-specified input-dependent rules, deterministic IID/OOD generation, an
exact branch-wise oracle and the paper's `7 -> 8 -> 8 -> 3` ReLU network. The
fine-tuning harness compares matched initializations for hard-local, STE,
black-box smoothing, certificate-only SAGE and geometry-aware SAGE. Every
constrained evaluation uses the exact lazy branch search; the neural network
never supplies the final feasible output by itself.

Added `test/test_phase9_cooling.jl`. Targeted tests passed `55/55`: rule
activation, deterministic generation, train-only normalization, oracle
feasibility, matched initialization and all five backward paths.

### 2026-11-14 - smoke benchmark retained

Command:

```powershell
$env:SAGE_PHASE9_PROFILE='smoke'
julia --project=. scripts/benchmark_phase9_cooling.jl
```

Observed one-seed result (`40/40/40` train/IID/OOD, 20 updates, `N=4`,
`rho=0.15`):

- maximum/mean-OOD active rules: `4 / 2.175`;
- maximum oracle violation: `6.94e-18`;
- all constrained methods: IID/OOD CSAT `1.0`;
- pretrained-exact/SAGE-Geometry/black-box OOD MSE:
  `0.07962 / 0.06414 / 0.07122`;
- black-box/certificate/geometry global searches: `80 / 16 / 16`;
- certificate and geometry global-search reduction: `80%`;
- geometry ESS: `3.902 / 4`;
- all smoke diagnostic gates passed.

The smoke result is not statistical evidence and is not reported as the Phase
9 conclusion. It only confirms that the application exercises overlapping
native disjunctions, all paths run end-to-end and the proposed pilot budget is
feasible.

### 2026-11-14 - pilot protocol and gates frozen before results

The final pilot uses five paired seeds, 250 training points, 500 IID and 500
OOD test points per seed, five pretraining epochs, one exact-projection
fine-tuning epoch capped at 250 updates, `N=8`, `rho=0.15`, and learning rate
`1e-3`. These are smaller than the paper's 2,000-point test splits, so this is
an application pilot rather than a paper-result reproduction.

Frozen acceptance gates:

1. zero unresolved, infeasible or unsafe oracle/training/evaluation outputs;
2. at least three simultaneously active rules somewhere and mean OOD active
   rules at least `1.5`;
3. SAGE-Geometry mean OOD MSE no worse than its pretrained-exact start;
4. certificate and geometry SAGE mean OOD MSE each at most `1.15x` matched
   black-box smoothing;
5. certificate and geometry global-search reductions at least `25%` and
   `15%`, respectively;
6. mean geometry effective sample size at least `0.70N`.

Failures will be retained. The gates will not be relaxed after observing the
five-seed pilot.

### 2026-11-14 - retained pilot failure and backend repair

The first frozen-protocol pilot stopped at seed 1, hard-local step 147/sample
100 because the center search was `:uncertified`: one complete branch QP
returned HiGHS `OTHER_ERROR`. Reproduction showed a feasible three-rule branch
(`cold`, `humid`, `demand_response`) for which HiGHS reported a maximum primal
infeasibility of `5.41e-5` after claiming QP optimality. The run was correctly
rejected; no approximate point was accepted and no gate was changed.

The repair is backend-level, not a cooling threshold adjustment. When a branch
QP returns neither `OPTIMAL` nor `INFEASIBLE`, `project_branch` now:

1. solves the identical feasible set as an LP;
2. accepts only an explicit LP `INFEASIBLE` result as an infeasibility
   certificate;
3. if the LP is feasible, solves the low-dimensional Euclidean projection by
   enumerating KKT active sets and re-runs the original logical validator;
4. otherwise leaves the branch unresolved.

The failing branch is now accepted only after the analytic result passes both
global original-logic and selected-branch checks. A regression test was added;
the targeted Phase 9 suite passes `58/58` after the repair. The frozen pilot
protocol and acceptance thresholds above remain unchanged.

### 2026-11-14 - frozen five-seed pilot result

The unchanged pilot protocol completed with `PHASE9_COOLING_PILOT=PASS`.

| method | IID MSE | OOD MSE | IID/OOD CSAT | train seconds | global searches | complete branch solves |
|---|---:|---:|---:|---:|---:|---:|
| base (raw network) | 0.15077 | 0.23612 | 0.0504 / 0.0604 | 0 | 0 | 0 |
| pretrained exact | 0.05085 | 0.06426 | 1.0 / 1.0 | 0 | 0 | 0 |
| hard-local | 0.02879 | 0.03878 | 1.0 / 1.0 | 6.79 | 0 | 3,138 |
| STE | 0.02224 | 0.03004 | 1.0 / 1.0 | 7.00 | 0 | 1,870 |
| black-box | 0.02985 | 0.04385 | 1.0 / 1.0 | 15.91 | 10,000 | 17,104 |
| SAGE-Cert | 0.02831 | 0.04287 | 1.0 / 1.0 | 12.08 | 3,353 | 18,722 |
| SAGE-Geometry | 0.02791 | 0.04178 | 1.0 / 1.0 | 13.02 | 3,496 | 20,367 |

Additional diagnostics:

- maximum/mean OOD active rules: `5 / 2.216`;
- maximum oracle violation: `1.11e-16`;
- SAGE-Geometry minus black-box paired OOD-MSE difference:
  `-0.00207`, 95% CI `[-0.00572, 0.00158]`;
- SAGE-Cert/SAGE-Geometry global-search reductions:
  `66.47% / 65.04%`;
- SAGE-Geometry mean ESS: `7.677 / 8`;
- safety, native-disjunction, learning, matched-quality, compute and ESS gates:
  all `PASS`.

## Interpretation boundary

This pilot shows that the proposal-aligned certificate and geometry pipeline
works end-to-end on input-dependent native disjunctions, preserves exact
logical feasibility and removes roughly two thirds of perturbed global
searches while matching the black-box estimator's mean OOD quality. It does
**not** establish a statistically significant OOD-MSE advantage: the paired
five-seed confidence interval includes zero. STE and hard-local are also
strong on this reconstruction, so Phase 9 is application readiness evidence,
not a claim that SAGE dominates every surrogate.

Because the official repository omits the cooling experiment's numeric
configuration, the result cannot be presented as an exact reproduction of the
paper table. A report must label it a paper-spec reconstruction and publish
`CoolingConfig`, seeds, all methods and both retained failure and repair.

### 2026-11-14 - packaging and regression

The Kaggle notebook now defaults to `RUN_MODE='phase9'`, writes the pilot CSV,
trajectory and JSON into the timestamped evidence bundle, and requires the
`PHASE9_COOLING_PILOT=PASS` token. The runner prints and serializes every
`CoolingConfig` field and prints progress after each seed/method so a Kaggle
CPU run is observable.

The first full-suite run exposed an ambiguous test binding because the Phase 8
and Phase 9 helper modules both export `prepare_problem`. This integration
failure was not algorithmic; Phase 9 tests were changed to use explicit module
qualification. The combined Phase 8/9 check then passed `58/58`, and the final
full regression passed:

```text
Test Summary:     | Pass  Total     Time
DisjunctiveNet.jl | 3160   3160  1m50.7s
```

A final smoke run after notebook/progress/config serialization changes emitted
`PHASE9_COOLING_SMOKE=COMPLETE`; all six diagnostic gates remained `PASS`.
