# Phase 12 research log — matched SAGE/STE/hard-local head-to-head

Date: 2026-11-14  
Status at registration: protocol and gates frozen before the Phase 12 pilot

## Why this phase exists

Phase 11 isolated the certified-multiscale mechanism and therefore compared
only black-box smoothing, fixed geometry SAGE and multiscale SAGE. It showed a
57.5% reduction in expensive global searches without a statistically
significant quality change, but it did not rerun the two cheap heuristic
backwards, hard-local and STE. Phase 12 closes that comparison gap. It is not a
new dataset and does not alter the Phase 10/11 Job-Shop model.

## Geometry versus multiscale note

Both SAGE variants target the same compact `q`, fixed physical radius
`rho=0.50`, exact feasible forward and Eq. (17)-(18) gradient. They differ only
in the importance proposal and allocation of solver work:

- `sage_geometry` uses optimizer-derived Eq. (19)-(21) boundary normals at the
  one target scale;
- `sage_multiscale` samples the exact mixture
  `w_base*q + w_local*p_local + w_geometry*p_focus`, where `p_local` is scaled
  by the certified Phase 3 margin and every sample receives ordinary `q/p`
  importance weight.

The multiscale proposal therefore changes neither the target objective nor the
forward solution. Base mass preserves support, certified-local samples exploit
stable regions, and focused samples estimate boundary switching. No gradient
is propagated through the certificate, normals or proposal parameters.

## Frozen comparison contract

Five methods are trained from the identical pretrained network for every seed:

1. `hard_local`: differentiate the center winning branch only;
2. `ste`: pass the exact projected-output loss gradient straight through;
3. `blackbox`: independent compact smoothing with exact perturbed searches;
4. `sage_geometry`: strengthened fixed-scale geometry proposal;
5. `sage_multiscale`: certified local/base/focused mixture.

The primary protocol matches fine-tuning updates, data, initialization,
minibatch order, `q`, `rho`, `N` and learning rate. A secondary wall-clock
replay uses the slowest primary method time within each seed as the common
budget. Method-specific replay step counts are frozen from measured primary
seconds-per-step, never from OOD quality, and are capped at five times the
primary updates. Actual replay seconds and overshoot are reported; this
hardware-dependent protocol is diagnostic rather than a mathematical gate.

## Frozen switching-exposure audit

OOD strata are fixed before method fine-tuning from the common pretrained
network. For each OOD instance, eight common antithetic target-`q` probes are
solved exactly and the fraction whose branch differs from the center winner is
recorded. The lower and upper halves of this score (deterministic index
tie-break) form `lower_switch` and `higher_switch` strata. The same instance
partition is then used for every method. This diagnostic never selects a
checkpoint, proposal or hyperparameter.

This rank split is intentionally called switching *exposure*, not certified
ground-truth boundary distance. It guarantees paired nonempty strata while the
raw switch rates reveal whether the intended difficulty contrast exists.

## Pilot configuration and pre-registered gates

The accepted pilot retains Phase 11's five seeds, 16 train instances, 12 IID
and 12 OOD instances, eight pretraining epochs, six fine-tuning updates,
`N=4`, `rho=0.50` and learning rate `1e-3`.

The Phase 12 token passes only when all of the following hold:

1. every exact constrained IID/OOD output has CSAT 1.0, maximum violation at
   most `1e-6`, and zero unresolved or logical failures;
2. the switching audit creates nonempty lower/higher strata for every seed and
   the aggregate higher-stratum probe rate is not below the lower rate;
3. multiscale OOD loss improves the exact pretrained OOD loss;
4. multiscale OOD loss is at most `1.10x` both STE and hard-local OOD loss
   (non-inferiority, not superiority);
5. multiscale global searches are at least 20% lower than matched black-box and
   at least 10% lower than fixed geometry;
6. multiscale complete branch solves are at most `1.05x` fixed geometry;
7. at least 50% of multiscale estimator calls have positive local radius and
   mean ESS is at least `0.40N`;
8. multiscale higher-switch OOD loss is at most `1.10x` both STE and hard-local
   higher-switch loss.

Paired 95% intervals against STE and hard-local are mandatory outputs. They
are interpretation fields, not gates; an interval crossing zero forbids a
superiority claim. Wall-clock replay quality is also reported without a pass
threshold because step allocation and timer granularity are platform
dependent.

No gate may be changed after observing the five-seed pilot. Smoke failures,
pilot failures and negative comparisons remain in this file.

## Implementation smoke retained before the pilot

The one-seed/two-update smoke completed both protocols and all exact forwards.
The common switching audit separated mean probe rates `0.333/0.667` for the
lower/higher halves. Matched-step results were:

| method | OOD loss | seconds | global searches | branch solves | ESS |
|---|---:|---:|---:|---:|---:|
| hard-local | 0.08287 | 0.322 | 0 | 6 | n/a |
| STE | 0.08122 | 0.291 | 0 | 4 | n/a |
| black-box | 0.08557 | 0.566 | 8 | 15 | n/a |
| fixed geometry | 0.07590 | 0.450 | 4 | 34 | 3.950 |
| multiscale | 0.07519 | 0.384 | 1 | 30 | 1.000 |

Maximum logical violation was `1.80e-16`. The smoke multiscale ESS was only
`1/4`, below the already frozen pilot threshold `0.40N=1.6`. Because a
two-update smoke is not the registered pilot, this result was retained and the
threshold was **not** lowered. The pilot was then run unchanged.

## Frozen five-seed result

The primary matched-step protocol passed every pre-registered gate:

| method | OOD loss | lower-switch loss | higher-switch loss | seconds | global searches | branch solves | ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard-local | 0.105711 | 0.113943 | 0.097478 | 0.656 | 0 | 68 | n/a |
| STE | 0.104546 | 0.114483 | 0.094609 | 0.635 | 0 | 38 | n/a |
| black-box | 0.111687 | 0.117718 | 0.105656 | 1.175 | 120 | 196 | n/a |
| fixed geometry | 0.107959 | 0.116805 | 0.099114 | 1.116 | 69 | 488 | 3.964 |
| multiscale | 0.107084 | 0.115872 | 0.098296 | 1.104 | 51 | 460 | 2.822 |

The shared pretrained audit produced mean switch rates `0.1042` and `0.5750`
for lower/higher exposure, confirming that the ranking formed a meaningful
difficulty contrast. Every method used 30 total updates over five seeds.

Multiscale-minus-STE OOD loss was `+0.002538`, 95% CI
`[-0.002174, 0.007251]`; multiscale-minus-hard-local was `+0.001373`, CI
`[-0.003180, 0.005926]`. On the higher-switch stratum the corresponding
differences were `+0.003687`, CI `[-0.002590, 0.009965]`, and `+0.000818`, CI
`[-0.004553, 0.006189]`. Every interval crosses zero. Phase 12 therefore
supports registered non-inferiority at this tolerance, not accuracy
superiority over either heuristic.

The Phase 11 computation result reproduced inside the complete comparison:
multiscale reduced global searches by `57.5%` versus black-box and `26.09%`
versus fixed geometry, used `0.943x` fixed geometry branch solves, had positive
local radius in `70%` of calls and retained ESS `2.822/4`. Maximum logical
violation was `2.22e-16`.

## Calibrated-time diagnostic and limitation

The hardware-dependent replay allocated approximately the slowest primary
method's time to each method. Across seeds, hard-local and STE received 56 and
57 updates, while black-box, fixed geometry and multiscale received 32 each.

| method | OOD loss | seconds | total updates |
|---|---:|---:|---:|
| hard-local | **0.094985** | 0.786 | 56 |
| STE | 0.095507 | 0.787 | 57 |
| black-box | 0.111565 | 1.185 | 32 |
| fixed geometry | 0.107068 | 1.217 | 32 |
| multiscale | 0.106533 | 1.099 | 32 |

This is a real negative boundary: when cheap heuristic backwards are allowed
to convert their speed into almost twice as many updates, they achieve lower
OOD loss on this small Job-Shop slice. Timer-based allocation is approximate
and actual seconds differ, so it is not a strict platform-independent
matched-time theorem. It nevertheless prevents presenting the matched-update
result as universal practical superiority.

The supported Phase 12 claim is narrower: certified multiscale SAGE preserves
exact feasibility, is statistically tied with STE/hard-local at matched
updates under five seeds, and materially reduces expensive global searches
relative to solver-based smoothing. Its principled gradient has not yet
outperformed the cheaper heuristic backwards under equal local wall time.

Final local token:

```text
PHASE12_HEAD_TO_HEAD=PASS
```

## Packaging and regression verification

- Moved the exact switching-exposure partition into the shared Job-Shop module
  so the benchmark and tests use one implementation.
- Added deterministic tests for exact audit feasibility, disjoint/complete
  shared strata, identical initial weights and all five backward paths.
- Added `phase12` Kaggle mode, made it the notebook default, included the gate
  in `full` mode and routed run, trajectory, switching-audit and JSON artifacts
  into the reproduction bundle.
- Updated the root README, research-log index and ignored local result folders.
- Targeted Phase 12 tests passed `15/15`.
- Full package regression passed `3496/3496` in `1m37.8s` on Julia 1.13.3.

Local pilot result folders remain ignored. Independent Kaggle execution is the
next reproduction step; its commit and platform timings must be logged rather
than silently replacing these local results.
