# Unified full Kaggle verification

Date: 2026-11-13

## Purpose

Run every accepted foundation, gradient, learning, multi-branch, and regression
gate in one fresh `RUN_MODE = "full"` notebook execution. This avoids combining
PASS claims from different code states and establishes a single pre-Phase-7
baseline.

## Executed revision and environment

- Git commit: `38c5e2658c2c59e319111a76f5855da1671a9b38`
- Julia: 1.13.3
- Python: 3.13.18
- Platform: Kaggle Linux x86-64
- CPU count: 4
- Reported memory: 32,869,296 KiB
- Accelerator requirement: CPU only

The Julia environment was already cached in the active Kaggle session, so
instantiate/precompile completed in `3.008 s`.

## Gate results

| Stage | Result | Elapsed time |
|---|---:|---:|
| Foundation verification | PASS | 90.886 s |
| SAGE gradient verification | PASS | 28.812 s |
| Controlled discontinuity learning | PASS | 21.889 s |
| Multi-branch matched-budget benchmark | PASS | 26.913 s |
| Full regression suite | 2671 / 2671 PASS | 130.060 s |

Required machine-readable tokens were present:

```text
FOUNDATION_VERIFICATION=PASS
SAGE_GRADIENT_VERIFICATION=PASS
DISCONTINUITY_LEARNING_GO_NO_GO=PASS
PHASE6_MATCHED_BUDGET=PASS
```

## Important observations

1. The DNF, CNF, and partial-DNF convexified baselines returned finite optimal
   solutions that were not feasible for the original disjunctive logic. The
   validator rejected the logical gap, while the exact branch-wise foundation
   output was logically feasible. This preserves the motivation for exact
   feasible forwards rather than treating a convexified optimum as ground
   truth.
2. The SAGE gradient stable/switching checks reproduced their accepted values,
   including absolute analytic gradient error `0.018797337172171913` and MC
   mean estimate `-10.319649960744337` against reference `-10.368`.
3. In the controlled scalar learning task, hard local differentiation never
   switched branches, while STE, black-box smoothing, and SAGE switched in all
   ten seeds. STE remained much cheaper in this simple one-dimensional case,
   so later experiments must retain it as a baseline.
4. Phase 6 reproduced exact/backend parity error
   `5.8347694962500474e-8`, 92.857% competitor pruning, 50% fewer global
   searches, and logical feasibility for every perturbation.

## Phase 6 timing replication

In this full run:

- black-box complete backward time: `0.65712942 s`;
- SAGE complete backward time: `0.581676864 s`;
- SAGE/black-box ratio: `0.8851785451943391`;
- measured SAGE improvement: approximately `11.48%`.

The direction of the controlled wall-time gain now agrees across the local run
(`17.14%`), the Phase-6-only Kaggle run (`12.02%`), and this full Kaggle run
(`11.48%`). The variation in magnitude confirms that the result is still
machine- and small-workload-sensitive. It is evidence to continue scaling
experiments, not evidence for a general speedup claim.

## Decision

All pre-Phase-7 gates pass on one revision, with no regression failures. Freeze
this result as the unified baseline and proceed to a pre-registered scaling
grid over dimension, branch count, smoothing radius, and sample budget. The
next phase must compare hard-local, STE, black-box, SAGE without pruning, and
certified SAGE under common random numbers and matched computational budgets.
