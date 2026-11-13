# Phase 6 - Multi-Branch Matched-Budget Benchmark

Date: 2026-11-13

## Proposal contract

This phase implements the first controlled slice of proposal experiments E1,
E2 and E3. It keeps the mathematical objective and estimator fixed:

- exact branch-wise Euclidean projection, proposal equations (7)-(10);
- certified margin and competitor pruning, equations (11)-(12);
- compact-support density `q(u) proportional to (1-norm(u)^2)^2`;
- fixed `rho` within every estimator call;
- reference branch held fixed inside the local derivative;
- local implicit term plus residual score correction from equations (17)-(18);
- no averaging of feasible outputs and no adaptive-radius objective change.

## Pre-registered benchmark

### A. Analytic/backend parity

A union of four separated axis-aligned boxes in two dimensions has an analytic
exact projection. The same 128 perturbations are passed to the analytic formula
and the certified Julia backend. Required parity error: `< 1e-6`; every
perturbed output must satisfy the original logical set.

### B. Gradient fidelity

At a fixed switching-sensitive center, black-box score smoothing and SAGE use
the same compact density, radius, 64 samples and common random numbers for 30
repetitions. A separate 100,000-sample proposal-faithful estimate is the
high-accuracy numerical reference for the same smoothed objective.

Required SAGE fidelity:

- mean cosine similarity at least `0.9`;
- relative bias at most `0.2`.

Bias, trace variance, RMSE, mean cosine and worst-seed cosine are reported for
both estimators. No result from a different surrogate objective is treated as
the reference.

### C. Certification and scalability slice

A union of eight separated boxes is evaluated at eight stable centers and
eight adjacent-boundary centers with the same `rho` and 12 perturbations per
center. Both methods see identical samples. The benchmark reports certificate
coverage, safe competitor pruning, global searches, local/reference branch
solves, lazy-search leaf/relaxation solves, complete backward wall time and
logical feasibility.

The Phase 6 gate requires:

1. correctness and fidelity above;
2. fewer perturbed global searches than black-box smoothing;
3. either lower trace variance or lower measured total backward wall time;
4. logically feasible outputs for every evaluated perturbation.

The third condition follows the proposal's Stage 3 go/no-go criterion: reduced
global-call counts alone are not sufficient evidence if neither statistical nor
total-compute efficiency improves.

## Reproduction

```text
julia --project=. scripts/benchmark_multibranch_matched_budget.jl
```

### Kaggle runner mode

`SAGE_Kaggle_Reproducibility.ipynb` now defaults to `RUN_MODE = 'phase6'`. A Kaggle **Run All** therefore installs/reuses the pinned Julia environment, checks out the latest repository, runs this Phase 6 gate, validates `PHASE6_MATCHED_BUDGET=PASS`, and bundles the logs. The already independently verified foundation, scalar-gradient, discontinuity-learning, and full-regression gates are marked `SKIPPED` in this resume mode rather than silently treated as passes.

Each execution writes to a timestamped `sage_reproduction_logs/<run-id>_<mode>/` directory so evidence from an older session cannot be mistaken for output from the current run.

Set `RUN_MODE = 'full'` in the first code cell when a complete paper/release reproduction is required. That mode reruns all gates and the full test suite.

Machine-searchable result:

```text
PHASE6_MATCHED_BUDGET=PASS|FAIL
```

## Recorded local result

Environment: Julia 1.13.3, HiGHS backend, Windows x86-64.

### A. Exact parity and feasibility

- analytic/backend gradient difference: `5.8347694962500474e-8`;
- switching perturbations: 36 / 128;
- all backend perturbed outputs passed original-logic feasibility.

### B. Matched-distribution gradient fidelity

Reference gradient from the independent 100,000-sample run was
`[-1.7105719603326774, -2.344762538785184]`.

| Estimator | Relative bias | Trace variance | RMSE | Mean cosine | Minimum cosine |
|---|---:|---:|---:|---:|---:|
| Black-box | 0.0154413 | 1.1134690 | 1.0561617 | 0.9710350 | 0.8134224 |
| SAGE | 0.0134904 | 1.0957776 | 1.0475260 | 0.9678456 | 0.8459258 |

SAGE's trace-variance ratio to black-box was `0.9841115`, a small `1.59%`
reduction. This is directionally positive but too small to support a broad
variance-reduction claim.

### C. Certified eight-branch workload

| Metric | Result |
|---|---:|
| Certificate rate | 0.50 |
| Safe competitor-pruning rate | 0.9285714 |
| Black-box perturbed global searches | 192 |
| SAGE perturbed global searches | 96 |
| Global-search reduction | 0.50 |
| SAGE reference / competitor branch solves | 192 / 96 |
| Black-box leaf / relaxation solves | 194 / 192 |
| Black-box wall time | 0.375442 s |
| SAGE complete backward wall time | 0.311101 s |
| SAGE / black-box wall-time ratio | 0.8286262 |

The measured SAGE backward was `17.14%` faster on this small warmed workload,
while preserving logical feasibility for every perturbation. Timing at this
scale is sensitive to machine noise and must be repeated on Kaggle and on larger
slices before it can support an empirical speed claim.

All three pre-registered gates passed:

```text
fidelity gate: PASS
global-search reduction gate: PASS
variance-or-wall-time efficiency gate: PASS
PHASE6_MATCHED_BUDGET=PASS
```

## Honest conclusion

The result establishes a first nontrivial multi-branch signal consistent with
the proposal: exact formula/backend agreement, high gradient fidelity, safe
competitor reduction, fewer global searches, and a small measured efficiency
gain. It does not yet establish scaling in dimension/branch count, robustness
of the timing gain, or superiority in end-to-end learning.

## Interpretation boundary

This is a controlled polyhedral slice, not a real-data or full training claim.
A pass supports moving to larger controlled slices and end-to-end Flux
integration. A failure identifies whether fidelity, variance, certification or
wall-clock overhead must be addressed before applications.

