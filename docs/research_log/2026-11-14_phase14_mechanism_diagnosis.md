# Phase 14 research log — objective-versus-variance diagnosis

Date: 2026-11-14  
Status at registration: mechanism grid and classification frozen before results

## Why another benchmark is not enough

Phase 13 found no optimization crossover. On 3x4 Job-Shop, multiscale SAGE was
closer than STE to a numerical smoothed-gradient reference but produced worse
task learning. Phase 14 does not search for another favorable application. It
tests whether the failure is caused primarily by the target smoothed objective
or by finite-sample SAGE variance.

## Proposal components retained from Đức

The underlying estimators are unchanged:

- exact feasible branch-wise forward;
- safe margin and competitor pruning, Eq. (11)-(12);
- compact target `q`, `beta=2`, fixed `rho` within each slice;
- full local/switching gradient, Eq. (17)-(18);
- optimizer-derived normals, Eq. (19)-(21);
- exact ordinary `q/p` for the certified base/local/focused proposal.

No backward, certificate, proposal weight or pruning rule is tuned in this
phase.

## New diagnostic additions and ownership boundary

The following are Phase 14 experimental diagnostics, not claims that they were
present in Đức's original proposal:

1. a cross-slice radius sweep `rho in {0.10, 0.30, 0.50}`; each estimator call
   still has one fixed target radius;
2. an SAGE sample sweep `N in {4, 8, 16, 32}`;
3. two independent 64-sample exact black-box gradients averaged into a
   numerical 128-sample reference, with replicate cosine reported;
4. a normalized output-space step of physical length `0.03` for direction-only
   comparison;
5. an independent 16-sample common-random estimate of the smoothed task
   objective before and after every candidate step.

These additions diagnose the existing method. They do not modify training and
must be described separately if used in a paper.

## Frozen pilot

The pilot uses the registered 3x4 Job-Shop family, five seeds, 16 training
instances, eight pretraining epochs and the first common training center.
Candidates are:

- high-sample numerical reference direction;
- hard-local;
- STE;
- multiscale SAGE at `N=4,8,16,32`.

For each candidate/radius the runner reports:

- cosine and relative error against the averaged reference;
- exact original projected-task loss improvement after the normalized step;
- common-random smoothed-objective improvement;
- whether the exact winner branch switches;
- estimator global searches, branch solves and ESS;
- exact logical violation.

No candidate chooses a step length or radius from observed quality. All rows
and all three radii remain in the output.

## Frozen diagnostic classification

With numerical tolerance `1e-5`, aggregate mean results determine two flags:

- `OBJECTIVE_MISMATCH`: at least two radii show positive reference-direction
  smoothed-objective improvement but non-positive original-task improvement;
- `ESTIMATOR_VARIANCE`: at least two radii show positive reference original-
  task improvement and SAGE-`N=32` improves original task by more than
  SAGE-`N=4` plus tolerance.

The final label is `BOTH`, `OBJECTIVE_MISMATCH`, `ESTIMATOR_VARIANCE`, or
`NEITHER_CONFIRMED`. This classification is diagnostic, not a pass condition.

## Frozen completion gates

`PHASE14_MECHANISM_DIAGNOSIS=PASS` requires only:

1. every exact center, stepped point and smoothed-evaluation output is feasible
   within `1e-6`, with no unresolved searches;
2. all `5 seeds x 3 radii x 7 candidates` rows are emitted;
3. both reference replicates and every candidate gradient are finite with
   positive reference norm;
4. all direction, original-improvement and smoothed-improvement metrics are
   finite;
5. every SAGE row has positive finite ESS and declared sample accounting;
6. reference-replicate cosine, solver work and branch-switch outcomes are
   reported without thresholding away a noisy reference.

Negative improvement, low reference agreement or failure to confirm either
mechanism remains a scientific result and cannot be converted into a software
failure by changing these rules.

## Smoke implementation check (before pilot)

The smoke profile completed before the pilot with one seed, radii `0.30/0.50`,
SAGE `N=4/8`, four samples per reference replicate and four independent
objective-evaluation samples. It passed completeness, safety, finite-metric,
reference and SAGE-accounting checks; the maximum logical violation was
`8.33e-17`.

As expected for only four samples, the two reference replicates were not a
scientifically usable target (cosines `-0.123` and `0.408`). This result is
recorded only as an implementation check. No frozen pilot configuration,
classification rule or gate was changed after observing it.

## Frozen pilot result

The unchanged pilot completed all 105 candidate rows with
`PHASE14_MECHANISM_DIAGNOSIS=PASS` and classified the failure mechanism as
`PHASE14_DIAGNOSIS=ESTIMATOR_VARIANCE`. The maximum exact logical violation
was `2.64e-16`; no completeness, safety, reference, finite-metric or SAGE
accounting gate failed.

Mean results over five seeds were:

| rho | method | cosine to reference | relative error | exact improvement | smoothed improvement | ESS |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 0.10 | reference | 1.000 | 0.000 | 0.00606 | 0.00592 | — |
| 0.10 | hard-local | 0.881 | 0.576 | 0.00692 | 0.00661 | — |
| 0.10 | STE | 0.735 | 1.209 | 0.00595 | 0.00574 | — |
| 0.10 | SAGE N=4 | 0.885 | 0.579 | 0.00691 | 0.00661 | 2.78 |
| 0.10 | SAGE N=32 | 0.891 | 0.497 | 0.00665 | 0.00663 | 20.93 |
| 0.30 | reference | 1.000 | 0.000 | 0.00583 | 0.00543 | — |
| 0.30 | hard-local | 0.828 | 0.659 | 0.00692 | 0.00584 | — |
| 0.30 | STE | 0.721 | 1.131 | 0.00595 | 0.00513 | — |
| 0.30 | SAGE N=4 | 0.724 | 0.803 | 0.00619 | 0.00538 | 2.59 |
| 0.30 | SAGE N=32 | 0.866 | 0.507 | 0.00655 | 0.00607 | 22.58 |
| 0.50 | reference | 1.000 | 0.000 | 0.00562 | 0.00506 | — |
| 0.50 | hard-local | 0.812 | 1.137 | 0.00692 | 0.00474 | — |
| 0.50 | STE | 0.720 | 1.467 | 0.00595 | 0.00433 | — |
| 0.50 | SAGE N=4 | 0.787 | 1.023 | 0.00578 | 0.00454 | 2.77 |
| 0.50 | SAGE N=32 | 0.872 | 0.731 | 0.00609 | 0.00511 | 22.51 |

The independent 64-sample reference-replicate cosines were `0.559`, `0.767`
and `0.625` for radii `0.10`, `0.30` and `0.50`. Thus even the numerical
reference remains noisy, but averaging the two declared replicates is much
more defensible than selecting one favorable draw.

The objective-mismatch vote was `0/3`: the reference direction improved both
the declared smoothed objective and the exact original loss at every radius.
The variance vote was `2/3`: at radii `0.30` and `0.50`, SAGE `N=32` improved
the exact loss more than SAGE `N=4` by the frozen tolerance. This supports a
finite-sample-variance bottleneck, not a claim that the smoothed objective is
misaligned.

## Interpretation and next decision

This is a mechanism diagnosis, not a SAGE superiority result. Hard-local had
the largest exact one-step improvement at every radius, while almost no
candidate changed the exact winning branch (only SAGE `N=8`, `rho=0.50`
switched in one of five seeds). The Phase 13 benchmark therefore spends much
of its optimization budget in a locally smooth regime where hard-local is
already strong and stochastic boundary estimation adds variance.

The next algorithmic phase should consequently be conditional rather than
globally increasing `N`: retain the Đức-derived exact forward, certificates,
normal geometry and exact `q/p`, but allocate boundary samples only when the
certificate/switching diagnostics predict useful boundary mass. Any adaptive
sample allocation, trigger or budget controller will be explicitly marked as
new work beyond the original proposal and must be pre-registered before its
quality result is observed.

## Packaging verification

- the public `estimate_output_gradient` diagnostic wrapper reuses the existing
  Job-Shop backward path and exposes solver work, proposal-component counts,
  ESS and failure accounting without changing network parameters;
- the targeted Phase 14 regression test passed `15/15`;
- the complete package suite passed `3537/3537` in `1m40.3s`;
- the Kaggle notebook defaults to standalone `RUN_MODE = "phase14"`, requires
  both the completion token and the frozen `ESTIMATOR_VARIANCE` diagnosis, and
  keeps Phase 14 out of the accepted-positive `full` bundle;
- generated smoke and pilot result directories are ignored, while the exact
  aggregate values and interpretation remain versioned in this log.
