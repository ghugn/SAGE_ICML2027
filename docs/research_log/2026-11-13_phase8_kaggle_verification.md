# Phase 8 - Kaggle reproduction of the UCI Iris pilot

Date: 2026-11-13

## Purpose

Reproduce the completed Phase 8 real-data pilot on Kaggle from the exact
committed revision. This run checks that the end-to-end training result,
proposal-aligned estimator, exact-feasibility checks and solver-work reduction
are not artifacts of uncommitted code or the local Windows environment.

This is a reproducibility record for the small Iris pilot. It does not turn the
pilot into evidence of general real-data superiority.

## Environment and revision

```text
git commit: 42ba497b5afcfd20a50dd0ccc9bc7a43f92ca63d
Julia:      1.13.3
Python:     3.13.18
platform:   Linux 6.18.55+, x86_64
CPUs:       4
memory:     32,869,296 KiB
run mode:   phase8
```

Package instantiation completed successfully in `3.184 s`. The Phase 8 stage
completed in `281.831 s` and emitted the required token
`PHASE8_IRIS_PILOT=PASS`.

Foundation, Phase 4-7 and the full regression suite were deliberately not
requested by the `phase8` resume mode. Their `SKIPPED` status is therefore not
a failure. Those gates were verified in their own recorded runs; this run
isolates the Phase 8 workload at the frozen revision above.

## Command and frozen design

```text
/kaggle/working/julia-1.13.3/bin/julia --project=. \
    scripts/benchmark_phase8_iris.jl
```

```text
dataset:                UCI Iris, DOI 10.24432/C56C76
task:                   sepal -> petal regression
disjunction:            train-only species boxes
seeds:                  5
steps/batch size:       60 / 12
SAGE/black-box samples: 8
rho/learning rate:      0.5 / 0.02
hidden width:           8
```

The run retained the proposal contract: exact feasible forward projection,
compact `q` with `beta=2`, fixed `rho`, fixed-reference local term and the
residual score correction in equations (17)-(18). Black-box and SAGE used
matched perturbation budgets.

## Aggregate result

| Method | Test MSE | Branch accuracy | Mean training seconds | Perturbed global searches | Sample certificates |
|---|---:|---:|---:|---:|---:|
| hard-local | 0.364828620 | 0.660000 | 4.211367 | 0 | 0 |
| STE | 0.281738757 | 0.740000 | 2.739162 | 0 | 0 |
| black-box | 0.291169702 | 0.726667 | 23.901154 | 28,800 | 0 |
| SAGE | **0.274588209** | 0.720000 | 15.971981 | 3,083 | 19,621 |

All pre-registered gates passed:

```text
analytic/backend parity:          PASS
feasibility and resolution:       PASS
learning from initialization:     PASS
matched black-box quality:        PASS
hard-local improvement:           PASS
global-search reduction:          PASS
end-to-end training wall time:    PASS
```

The exact-forward parity errors remained small:

```text
maximum output parity error:      7.0516677e-8
maximum objective parity error:   2.4863009e-15
```

Every evaluated output passed the original logical-feasibility validator and
no training sample was unresolved.

## Computation result

SAGE reduced perturbed global searches from `28,800` to `3,083`, a reduction
of `89.295%`. The safe pointwise certificates resolved another `19,621`
samples without a perturbed global search. The mean end-to-end training-time
ratio was:

```text
SAGE / black-box = 0.6682514623
```

Thus SAGE was approximately `33.2%` faster than matched black-box smoothing on
this Kaggle run. Absolute times were slower than the local run, but the
quality metrics, search counts and relative timing were reproduced. The
search reduction is the more implementation-independent efficiency result.

## Statistical interpretation

The five-seed paired intervals were:

```text
SAGE - black-box test MSE:
  mean = -0.0165814930, 95% CI [-0.0582501989, 0.0250872128]

SAGE - hard-local test MSE:
  mean = -0.0902404110, 95% CI [-0.2269993239, 0.0465185019]

SAGE - black-box branch accuracy:
  mean = -0.0066666667, 95% CI [-0.0251733333, 0.0118400000]
```

SAGE achieved the best mean test MSE, but all three intervals include zero.
The defensible claim is therefore that SAGE matched black-box predictive
quality while preserving exact feasibility and using substantially less exact
global-search work. This five-seed Iris pilot does **not** establish a
statistically significant quality advantage. STE also remains an important
low-cost comparator: it had the best mean branch accuracy and was much faster,
while lacking the exact proposal estimator and solver-work interpretation.

## Decision

`PHASE8_IRIS_PILOT=PASS` independently reproduces the first real-data
end-to-end gate. Phase 8 can be frozen as a pipeline-validation result. The
next empirical phase should use a larger task with native disjunctive
constraints, more branches and higher-dimensional geometry, while retaining
matched budgets, exact-feasibility validation, multiple seeds and confidence
intervals.

## Saved raw evidence

The Kaggle session stored:

```text
sage_reproduction_logs/20261113_164232_phase8/
  phase8_iris_results/phase8_runs.csv
  phase8_iris_results/phase8_trajectories.csv
  phase8_iris_results/phase8_summary.json
  final_summary.json
```
