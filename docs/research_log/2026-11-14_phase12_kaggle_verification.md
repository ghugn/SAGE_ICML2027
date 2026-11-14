# Phase 12 Kaggle verification

Date: 2026-11-14  
Commit: `e99ea8bc56ab62d7abf112dfb9feda238c73a427`  
Platform: Kaggle Linux x86_64, 4 CPU, Julia 1.13.3

## Reproduction command and status

The default notebook `phase12` mode ran:

```text
/kaggle/working/julia-1.13.3/bin/julia --project=. \
  scripts/benchmark_phase12_head_to_head.jl
```

Package instantiation completed with exit code zero in `705.389s`. The Phase
12 stage completed with exit code zero in `339.506s`, found its required token
and reported:

```text
PHASE12_HEAD_TO_HEAD=PASS
```

All earlier phases and the full regression suite were `SKIPPED` intentionally
because `RUN_MODE="phase12"` runs only this accepted gate. This is not evidence
that an earlier gate failed; their accepted outputs remain separately logged.

## Matched-step reproduction

Every method used 30 total updates over the same five seeds. The deterministic
metrics reproduced the local pilot to numerical precision:

| method | OOD loss | lower-switch loss | higher-switch loss | seconds | searches | branch solves | ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard-local | 0.105711 | 0.113943 | 0.097478 | 1.371 | 0 | 68 | n/a |
| STE | 0.104546 | 0.114483 | 0.094609 | 1.389 | 0 | 38 | n/a |
| black-box | 0.111687 | 0.117718 | 0.105656 | 2.385 | 120 | 196 | n/a |
| fixed geometry | 0.107959 | 0.116805 | 0.099114 | 2.316 | 69 | 488 | 3.964 |
| multiscale | 0.107084 | 0.115872 | 0.098296 | 2.217 | 51 | 460 | 2.822 |

The common pretrained switching audit again gave lower/higher exposure rates
`0.10417/0.575`. Multiscale reduced global searches by `57.5%` versus
black-box and `26.09%` versus fixed geometry, used `0.943x` fixed geometry
branch solves, and had positive certified local radius in `70%` of estimator
calls. Maximum logical violation was `3.33e-16`.

The paired multiscale-minus-STE OOD difference was `+0.002538`, 95% CI
`[-0.002174, 0.007251]`; multiscale-minus-hard-local was `+0.001373`, CI
`[-0.003180, 0.005926]`. Higher-switch intervals also crossed zero. Kaggle
therefore independently reproduces non-inferiority under the registered 10%
margin and does not establish accuracy superiority.

## Platform-dependent calibrated-time replay

The timer-derived step allocation changed on Kaggle, as pre-registered:

| method | OOD loss | mean seconds | total updates |
|---|---:|---:|---:|
| hard-local | **0.096587** | 1.533 | 51 |
| STE | 0.097697 | 1.487 | 50 |
| black-box | 0.111687 | 2.364 | 30 |
| fixed geometry | 0.107959 | 2.170 | 30 |
| multiscale | 0.106904 | 2.177 | 31 |

This reproduces the local qualitative limitation: cheap heuristic backwards
convert their lower per-update cost into substantially more updates and lower
OOD loss. Actual seconds are not exactly equal because allocation uses primary
seconds-per-step and training stops only at update boundaries; the replay is a
hardware diagnostic, not a strict timing theorem.

## Supported conclusion

Kaggle independently verifies the Phase 12 mechanism and comparison on the
exact committed revision. Certified multiscale SAGE is exact-feasible,
statistically tied with hard-local and STE at matched updates, and more
search-efficient than solver-based black-box/fixed-geometry smoothing. It is
not yet more wall-clock efficient than the heuristic backwards on this small
Job-Shop slice. This limitation remains part of the research record.

Artifacts were written under:

```text
/kaggle/working/sage_reproduction_logs/20261114_122959_phase12/
```
