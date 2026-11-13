# Phase 7 - Kaggle reproduction

Date: 2026-11-13

## Purpose

Reproduce the completed Phase 7 controlled scaling gate outside the local
Windows environment, using the committed Kaggle notebook and an exact Git
revision. This checks that the result is not dependent on uncommitted code,
stale local output, or the development machine.

## Environment and revision

```text
git commit: 96a3f74e84507247aadbd26db01b59277169bac2
Julia:      1.13.3
platform:   Linux 6.18.55+, x86_64
CPUs:       4
memory:     32,869,296 KiB
run mode:   phase7
```

The notebook deliberately skipped foundation, Phase 4-6, and full-regression
stages in `phase7` resume mode. Those `SKIPPED` values mean that the stages were
not requested in this run; they are not failures. The completed Phase 7 script
was required to emit the exact token `PHASE7_COMPLETE=PASS`.

## Command

```text
/kaggle/working/julia-1.13.3/bin/julia --project=. \
    scripts/benchmark_phase7_complete.jl
```

Instantiation took `743.327 s`, primarily due to first-session Julia package
download/precompilation. The Phase 7 benchmark itself took `36.714 s`.

## Result

```text
PHASE7_COMPLETE=PASS
[05_phase7_complete] exit=0 elapsed=36.714s token_ok=True
```

All acceptance sub-gates passed:

```text
reference convergence:          PASS
gradient fidelity:              PASS
per-configuration RMSE:         PASS
analytic/backend parity:        PASS
safe pruning and feasibility:   PASS
global-search reduction:        PASS
variance efficiency:            PASS
wall time:                      PASS
proposal variance-or-time:      PASS
```

The statistical values exactly reproduced the deterministic seeded local run:

```text
aggregate SAGE/black-box variance ratio: 0.7196172875
aggregate SAGE/black-box RMSE ratio:     0.8446540492
unsafe pruning count:                    0
maximum projection parity error:         3.3941122e-7
maximum gradient parity error:           2.3710520e-6
black-box/SAGE global searches:          96 / 48
global-search reduction:                 50%
```

Kaggle timing was:

```text
median black-box time:          0.114928833 s
median SAGE time:               0.114479564 s
SAGE/black-box wall-time ratio: 0.996090894
```

## Interpretation

The Kaggle wall-time sub-gate passes numerically, but the measured advantage is
only about `0.39%`. Local completed runs produced ratios between `1.020` and
`1.068`. Therefore the defensible conclusion is **backend time is currently at
parity within run-to-run and platform variation**, not that SAGE has established
a meaningful speedup.

The robust positive evidence is instead:

- exact feasible outputs and zero observed unsafe pruning;
- 50% fewer global-search invocations;
- approximately 28% lower aggregate estimator variance;
- approximately 15.5% lower aggregate RMSE;
- fidelity maintained through `d=8,K=16` with the declared dimension-aware
  sample budget.

This reproduction is sufficient to freeze Phase 7 and design a real-data
pilot. It is not a real-data result and does not establish end-to-end training
speed or task-quality superiority.

## Saved raw evidence

The Kaggle run stored the following files in its timestamped evidence bundle:

```text
sage_reproduction_logs/20261113_152942_phase7/
  phase7_complete_results/phase7_reference_convergence.csv
  phase7_complete_results/phase7_fidelity.csv
  phase7_complete_results/phase7_workload.csv
  phase7_complete_results/phase7_summary.json
  final_summary.json
```
