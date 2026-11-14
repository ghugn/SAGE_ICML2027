# SAGE Core 0.2.0-alpha.1 release verification

Date: 2026-11-14

Status: local release **PASS**; an independent Kaggle alpha-mode rerun remains
required before calling the tagged alpha externally reproduced.

## Release boundary

This alpha packages the exact-feasible SAGE research API, safe certificates and
the Component-4 geometry-aware importance sampler. It does not replace the
existing Flux layer's automatic backward pass and does not claim broad task
superiority. The public boundary and limitations are recorded in
`docs/SAGE_ALPHA.md`; the version is `0.2.0-alpha.1`.

## Commands and results

All commands used Julia 1.13.3 with the committed project environment.

```text
julia --project=. scripts/benchmark_sage_novelty_alpha.jl
SAGE_ALPHA_NOVELTY=PASS

julia --project=. test/runtests.jl
3102 / 3102 tests PASS

julia --project=. scripts/benchmark_multibranch_matched_budget.jl
PHASE6_MATCHED_BUDGET=PASS

julia --project=. scripts/benchmark_phase7_complete.jl
PHASE7_COMPLETE=PASS

julia --project=. scripts/benchmark_phase8_iris.jl
PHASE8_IRIS_PILOT=PASS
```

Phase 8 reproduced the committed quality values (`0.274588` SAGE mean test MSE
versus `0.291170` matched black-box), reduced perturbed global searches from
28,800 to 3,083 and used `0.647x` the local black-box training time. Its paired
quality confidence interval still includes zero and remains a pilot result.

## Note audit

The research-log index contains separate records for Phase 0 baseline, Phase 1
exact oracle, Phase 2 lazy search, Phase 3 switching certificate, Phase 4 SAGE
estimator, Phase 5 discontinuity learning, Phase 6 multi-branch benchmarking,
both the failed and completed Phase 7 scaling gates, Phase 8 Iris, Kaggle
verifications, workspace migration/consolidation and this alpha novelty gate.
Negative results have not been deleted or rewritten as passes.

## Remaining external verification

The Kaggle notebook now defaults to `RUN_MODE = 'alpha'` and writes the novelty
CSV/JSON evidence into the reproduction bundle. The tag `v0.2.0-alpha.1`
identifies the packaged code, but a Kaggle alpha run should still be recorded so
its final summary names the exact tag commit. Until then the alpha is locally
verified, not independently reproduced.
