# Phase 7 - Dimension and branch-count scaling smoke gate

Date: 2026-11-13

## Hypothesis

The proposal-faithful SAGE estimator should preserve exact feasible forwards,
safe competitor pruning, and acceptable gradient fidelity as dimension and
branch count increase. Certificate/pruning savings must translate into lower
complete backward time; reduced global-search counts alone are not sufficient.

## Mathematical contract kept fixed

- exact branch-wise Euclidean projection for every forward output;
- compact-support `q(u) proportional to (1 - norm(u)^2)^2` (`beta = 2`);
- one fixed `rho` within each stable or switching slice;
- one fixed reference branch inside the local implicit derivative;
- proposal equations (17)-(18): local implicit term plus residual score term;
- no averaging of feasible projected outputs;
- certificate pruning accepted only when an analytic geometric bound proves it
  safe;
- common random numbers and matched `q`, `rho`, and `N` for SAGE and black-box
  comparisons.

## Implementation

Added:

- `scripts/phase7_scaling_common.jl`: deterministic separated-box family,
  analytic oracle, estimator summaries, black-box backend, and the SAGE
  no-certificate ablation;
- `scripts/benchmark_phase7_scaling.jl`: smoke-grid runner with CSV/JSON output
  and machine-readable gate;
- `test/test_phase7_scaling.jl`: deterministic geometry, backend/oracle parity,
  equation (17)-(18) parity, feasibility, and stable-certificate tests;
- Kaggle notebook `RUN_MODE = "phase7"` support.

The methods compared are hard local, STE, black-box smoothing, SAGE without
certificate/pruning, and full certified SAGE. Certificate and no-certificate
SAGE share the same mathematical estimator; only their computation differs.

## Pre-registered smoke grid

```text
dimensions:         2, 4, 8
branches:           4, 8, 16
repetitions:        5
samples/estimate:   64
reference samples:  20,000
backend samples:    8 per center
workload centers:   2 stable + 2 switching per configuration
```

Acceptance was fixed before the full run:

1. no logical violations and no unsafe pruning;
2. projection parity below `1e-6` and backend-gradient parity below `1e-5`;
3. SAGE mean cosine at least `0.90` and relative bias at most `0.20` in every
   configuration;
4. SAGE RMSE no more than `10%` above matched black-box RMSE;
5. at least `25%` fewer global searches for `K >= 8`;
6. aggregated complete SAGE backward time no greater than black-box time for
   `K >= 8`.

## Exact command

```text
julia --project=. scripts/benchmark_phase7_scaling.jl
```

Outputs are written to `phase7_results/phase7_fidelity.csv`,
`phase7_results/phase7_workload.csv`, and
`phase7_results/phase7_summary.json`. Generated result directories are ignored
by Git; the numerical evidence is recorded below and Kaggle bundles retain the
raw files.

## Unit verification

The Phase 7-specific tests passed `62 / 62`:

```text
Phase 7 deterministic scaling family: 38 / 38
Phase 7 analytic oracle matches certified backend: 12 / 12
Phase 7 unpruned and certified SAGE preserve equation 17-18: 12 / 12
```

After integration, the complete package regression suite passed
`2733 / 2733` tests on Julia 1.13.3 in `91.6 s` (the previous 2671 tests plus
the 62 new Phase 7 tests).

## Pre-registered smoke result

Overall result:

```text
PHASE7_SCALING_SMOKE=FAIL
```

Correctness and structural results remained strong:

- maximum projection parity error: `3.394112215049316e-7`;
- maximum backend-gradient parity error: `6.7319754553134815e-6`;
- unsafe pruning count: `0`;
- every evaluated output passed original-logic feasibility;
- large-case global searches: black-box `192`, SAGE `96`;
- global-search reduction: `50%`.

Gradient fidelity by configuration:

| d | K | SAGE relative bias | SAGE RMSE | SAGE mean cosine | Black-box RMSE |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.1457 | 1.6030 | 0.9663 | 1.8235 |
| 2 | 8 | 0.1561 | 1.1780 | 0.9714 | 1.6167 |
| 2 | 16 | 0.1490 | 2.6683 | 0.9637 | 3.9755 |
| 4 | 4 | 0.2019 | 3.4393 | 0.8945 | 3.4470 |
| 4 | 8 | 0.1625 | 3.4307 | 0.8920 | 2.8846 |
| 4 | 16 | 0.1687 | 3.3357 | 0.8896 | 2.6646 |
| 8 | 4 | 0.5100 | 7.1338 | 0.8245 | 7.0561 |
| 8 | 8 | 0.4301 | 8.0797 | 0.7631 | 6.3775 |
| 8 | 16 | 0.2733 | 6.1745 | 0.8273 | 6.4045 |

The fidelity and RMSE gates failed. Fixed `N = 64` does not preserve estimator
quality as dimension grows; black-box fidelity also degrades, showing that the
problem is not a correctness discrepancy between formula and backend.

Complete large-case backward timing also failed:

```text
black-box time:       0.5684701 s
full SAGE time:       1.6806592 s
SAGE/black-box ratio: 2.95646
```

The stable half of the workload was always certified and required zero
perturbed global searches. In switching slices, however, the current sufficient
certificate became conservative in higher dimensions. Full SAGE then paid for
`N` fixed-reference branch projections plus many relevant competitor solves.
For `d=8, K=16`, the four-center aggregate used 32 reference solves and 240
competitor solves, despite halving the number of global-search invocations.
This explains why counting global searches alone overstated efficiency.

## Diagnostic sample-budget run

A non-gating diagnostic increased the budget to 10 repetitions and `N = 256`
without changing `q`, `rho`, or the estimator. Mean cosine exceeded `0.90` in
all configurations and the RMSE gate passed, but relative bias for `d=8,K=16`
remained `0.3666`; the fidelity gate therefore still failed. Timing remained
approximately `3.03x` black-box because backend samples and certificate geometry
were unchanged.

This diagnostic supports a dimension-dependent sample-complexity problem, but
does not justify changing the pre-registered result or claiming that `N = 256`
solves it.

## Decision and honest interpretation

Phase 7 is a useful negative result. The implementation remains exact and safe,
and the 2D signal from Phase 6 survives. The current estimator/budget and
certificate implementation do not yet scale efficiently to `d=4/8` on this
family. Do not proceed to a real-data superiority claim or hide the failure by
relaxing thresholds.

The next engineering/research step is Phase 7b, keeping the proposal objective
unchanged while addressing:

1. dimension-aware sampling and variance reduction under the same compact `q`;
2. certificate tightness near switching sets in higher dimensions;
3. reuse/batching of fixed-reference branch solves;
4. reporting branch solves and total time, never only global-call counts;
5. continued comparison with STE, which remains a strong low-cost heuristic on
   this controlled box family.

## Subsequent work

The smoke result above remains unchanged. Its bottlenecks were addressed in the
separate [`2026-11-13_phase7_completion.md`](2026-11-13_phase7_completion.md)
experiment, which adds tighter safe geometry, exact lazy incumbent reuse,
independent reference checks and dimension-aware antithetic sampling. Keeping
the experiments separate prevents the later successful gate from overwriting
this pre-registered negative result.
