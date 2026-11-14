# Phase 13 research log — pre-registered difficulty/crossover map

Date: 2026-11-14  
Status at registration: grid, metrics and interpretation frozen before results

## Motivation

Phase 12 showed that certified multiscale SAGE is statistically tied with STE
and hard-local at six matched updates, but the cheap heuristics win when their
per-update speed is converted into more updates. A harder task must not be
selected after looking for a favorable SAGE result. Phase 13 therefore reports
the complete registered grid and asks whether a crossover emerges as branch
count, switching exposure and training horizon increase.

## Architecture contract retained from Đức's proposal

Every slice retains:

1. exact feasible branch-wise forward projection;
2. Phase 3 distance margin and safe competitor pruning, Eq. (11)-(12);
3. one compact target `q`, `beta=2`, and one fixed `rho=0.50` per slice;
4. the complete local plus switching estimator, Eq. (17)-(18);
5. optimizer-derived normals, Eq. (19)-(21);
6. certified base/local/focused proposal with ordinary exact `q/p`, division by
   declared `N`, and no gradient through certificates or proposal parameters.

The target objective is not changed with difficulty. SAGE, STE and hard-local
start from the identical pretrained network, data order and optimizer state for
each seed/slice.

## Frozen grid

The controlled native-disjunction Job-Shop family is generalized without
changing its semantics:

| jobs x machines | operations | disjunctions | complete branches |
|---|---:|---:|---:|
| 3x3 | 9 | 9 | 512 |
| 3x4 | 12 | 12 | 4,096 |

Both sizes use five seeds, 16 train, 12 IID and 12 OOD instances, eight
pretraining epochs, `N=4`, `rho=0.50`, Adam learning rate `1e-3`, and matched
horizons `6` and `24` updates. Methods are `hard_local`, `ste`, and
`sage_multiscale`. The entire 2x2x3 grid is reported; no cell may be removed
after observation.

The hardest registered cell (`3x4`, 24 updates) additionally receives a
calibrated-time replay. Its step allocation is computed only from primary
seconds-per-step, capped at four times 24 updates, and never uses quality.

## Gradient and switching diagnostics

Before fine-tuning, every seed/size gets:

- a shared OOD switching-exposure audit using eight exact antithetic target-`q`
  probes per instance;
- one common training center gradient audit;
- a 64-sample antithetic exact black-box smoothing gradient as the reference;
- hard-local and STE deterministic gradients plus a 32-sample multiscale SAGE
  estimate compared by cosine and relative error.

The high-sample black-box estimate is a numerical reference for the declared
smoothed objective, not analytic ground truth. Its cost and randomness are
reported. It is used only for diagnosis, never for training or checkpoint
selection.

## Frozen outcomes and gates

`PHASE13_DIFFICULTY_MAP=PASS` means the registered map completed correctly,
not that SAGE won. Required gates are:

1. every constrained IID/OOD output is exactly feasible within `1e-6`, with
   zero unresolved/logical failures;
2. both registered sizes have the declared operations, disjunctions and branch
   counts, and their MILP oracle outputs pass original logic;
3. every method completes every registered seed/size/horizon cell from shared
   initialization;
4. switching audit strata are nonempty and higher exposure is not below lower
   exposure;
5. all gradient-audit vectors and metrics are finite, with positive reference
   norm and positive multiscale ESS;
6. multiscale retains at least 40% of `N` mean ESS and at least 30% positive
   certified-local calls in every size/horizon aggregate;
7. full loss, higher-switch loss, seconds, solver work, gradient alignment and
   learning-curve records are emitted for every cell.

The pre-registered crossover label is separate:

- `OBSERVED` only if SAGE has lower mean higher-switch OOD loss than both STE
  and hard-local in the `3x4,24` matched-update cell and at least one paired
  five-seed 95% interval excludes zero in SAGE's favor;
- otherwise `NOT_OBSERVED`.

Matched-time results cannot upgrade this label, but they can expose a practical
efficiency limitation. No loss-ratio gate is used to turn a negative scientific
answer into a software failure.

## Retained implementation smoke before the pilot

The one-seed smoke ran both `3x3` and `3x4`, horizons `2/4`, all three
backwards, the exact switching audit, gradient audit and hardest-cell
calibrated-time replay. Structure, safety, completion, stratification and
gradient-audit gates passed. Maximum logical violation was `2.22e-16`.

The generalization produced exactly `60` inputs, `12` outputs, `12`
disjunctions and `4,096` branches for `3x4`. No backend or MILP oracle was
replaced.

The frozen estimator gate failed in smoke because the one `3x4` seed had zero
positive certified-local calls at both horizons:

| size/horizon | SAGE ESS | positive local calls | local samples |
|---|---:|---:|---:|
| 3x3 / 2 | 2.461/4 | 1/2 | 3 |
| 3x3 / 4 | 3.223/4 | 1/4 | 3 |
| 3x4 / 2 | 3.963/4 | 0/2 | 0 |
| 3x4 / 4 | 3.965/4 | 0/4 | 0 |

This may indicate that the safe certificate becomes too conservative as the
coupled disjunction count grows. The registered `30%` local-call threshold is
not changed. The five-seed pilot will determine whether this is a single-seed
smoke artifact or a reproducible scaling boundary. If it persists, Phase 13 is
recorded as a negative diagnostic rather than relabeled after observation.

Smoke token:

```text
PHASE13_DIFFICULTY_MAP_SMOKE=COMPLETE
```

## Frozen five-seed pilot result

The complete registered grid ran without unresolved samples and passed safety,
structure, completion, stratification, gradient-audit and estimator gates.
Maximum logical violation was `2.64e-16`. The scientific crossover label was:

```text
NOT_OBSERVED
```

Matched-update results were:

| size/horizon | method | OOD loss | higher-switch loss | learning AUC | seconds | global searches | branch solves | ESS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3x3 / 6 | hard-local | 0.10571 | 0.10205 | 0.11178 | 0.729 | 0 | 68 | n/a |
| 3x3 / 6 | STE | **0.10455** | **0.09921** | **0.11120** | 0.669 | 0 | 38 | n/a |
| 3x3 / 6 | SAGE | 0.10708 | 0.10276 | 0.11247 | 1.010 | 51 | 460 | 2.822 |
| 3x3 / 24 | hard-local | 0.07882 | 0.07809 | 0.09591 | 2.636 | 0 | 289 | n/a |
| 3x3 / 24 | STE | **0.07525** | **0.07377** | **0.09528** | 2.725 | 0 | 172 | n/a |
| 3x3 / 24 | SAGE | 0.08293 | 0.07992 | 0.09882 | 4.426 | 262 | 1,979 | 3.053 |
| 3x4 / 6 | hard-local | 0.14775 | 0.14433 | 0.15473 | 1.292 | 0 | 74 | n/a |
| 3x4 / 6 | STE | **0.14375** | **0.14177** | **0.15273** | 1.464 | 0 | 47 | n/a |
| 3x4 / 6 | SAGE | 0.14812 | 0.14488 | 0.15492 | 2.175 | 61 | 596 | 2.817 |
| 3x4 / 24 | hard-local | 0.10748 | 0.10661 | 0.13422 | 5.598 | 0 | 321 | n/a |
| 3x4 / 24 | STE | **0.09953** | **0.09653** | **0.12813** | 6.759 | 0 | 201 | n/a |
| 3x4 / 24 | SAGE | 0.11154 | 0.11082 | 0.13530 | 9.541 | 289 | 2,454 | 3.183 |

On the hardest cell, SAGE-minus-STE higher-switch loss was `+0.014289`,
95% CI `[0.003030, 0.025548]`: STE was significantly better. SAGE-minus-
hard-local was `+0.004208`, CI `[-0.003833, 0.012249]`. Increasing branch
count and training horizon therefore did not reveal the registered crossover;
the gap against STE widened.

## Mechanism diagnostics

Switching exposure was real rather than nominal:

| size | lower exposure | higher exposure |
|---|---:|---:|
| 3x3 | 0.14 | 0.61 |
| 3x4 | 0.13 | 0.52 |

The smoke local-coverage failure did not persist across five seeds. SAGE
positive-local rates were `70.0%`, `60.8%`, `63.3%`, and `49.2%` for
`3x3/6`, `3x3/24`, `3x4/6`, and `3x4/24`; all passed the frozen 30% gate.

The 64-sample black-box gradient reference audit gave these mean
cosine/relative-error values:

| size | hard-local | STE | SAGE multiscale |
|---|---:|---:|---:|
| 3x3 | 0.67 / 0.74 | **0.82 / 0.67** | 0.77 / **0.67** |
| 3x4 | 0.73 / 0.75 | 0.70 / 1.20 | **0.74 / 0.69** |

At `3x4`, SAGE is closer to the numerical smoothed-gradient reference than
STE, yet trains worse. This falsifies the simple explanation that STE wins only
because the benchmark is too easy for its biased gradient. Plausible remaining
causes are the mismatch between the fixed smoothed objective and finite-step
task improvement, SAGE estimator variance at training `N=4`, and the optimizer
benefit of STE's direct task gradient. These are hypotheses, not post-hoc
repairs.

## Calibrated-time hardest-cell limitation

The timer replay preserved the same ordering:

| method | OOD loss | higher-switch loss | mean seconds | total updates |
|---|---:|---:|---:|---:|
| hard-local | 0.09348 | 0.09029 | 12.54 | 205 |
| STE | **0.08947** | **0.08524** | 10.47 | 174 |
| SAGE | 0.11154 | 0.11082 | 9.37 | 120 |

Timing and allocated steps vary by solver path and platform, so this remains a
diagnostic. It nevertheless supplies no wall-clock advantage for SAGE.

## Supported conclusion

Phase 13 is a successful difficulty-map experiment with a negative crossover
answer. Harder native disjunctions and longer training do not make SAGE beat
STE; on the hardest higher-switch stratum, STE is significantly better. The
mathematical architecture remains exact and the gradient-fidelity mechanism is
measurable, but fidelity to the fixed smoothed objective has not translated
into superior optimization on this Job-Shop family. Further work should study
the objective/variance mechanism directly rather than search for another task
where SAGE happens to win.

Final local tokens:

```text
PHASE13_DIFFICULTY_MAP=PASS
PHASE13_CROSSOVER=NOT_OBSERVED
```

## Packaging and regression verification

- Generalized the shared Job-Shop generator, feature/target tensors and MLP
  dimensions while protecting the controlled envelope at 18 disjunctions.
- Added a reusable exact `output_gradient_audit` and retained the shared
  switching-exposure partition from Phase 12.
- Added deterministic tests for 3x4 dimensions, 4,096-branch structure, MILP
  oracle feasibility, finite proposal-faithful gradient audit and rejection of
  configurations beyond the controlled branch envelope.
- Added standalone `phase13` Kaggle mode and made it the notebook default. It
  requires both the map PASS token and the frozen negative crossover token;
  Phase 13 is deliberately excluded from `full` because it is a long negative
  diagnostic.
- Targeted Phase 13 tests passed `26/26`.
- Full package regression passed `3522/3522` in `1m42.0s` on Julia 1.13.3.
- Notebook JSON parsing and the post-packaging Phase 13 smoke both passed.

Generated smoke/pilot folders remain ignored. The next required action is an
independent Kaggle run of the committed revision; platform timing and any
cross-platform metric drift must be logged without rewriting this local result.
