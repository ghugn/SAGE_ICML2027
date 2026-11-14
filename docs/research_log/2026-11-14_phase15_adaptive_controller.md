# Phase 15 — Certificate-triggered adaptive SAGE controller

Date: 2026-11-14

## Motivation from the frozen evidence

Phase 13 found no training crossover against hard-local or STE on the 3x4
Job-Shop family. Phase 14 then rejected the registered objective-mismatch
diagnosis (`0/3` radius votes) and supported a finite-sample-variance diagnosis
(`2/3` votes): `N=32` improved the exact one-step loss more than `N=4` at
`rho=0.30/0.50`. At the same time, almost every normalized step retained the
same exact branch and hard-local was strongest. Globally replacing `N=4` by
`N=32` would therefore spend solver work at centers where boundary estimation
is unlikely to help.

Phase 15 tests a conditional controller: use the minimum SAGE batch when
stability is proved, and otherwise allocate a larger SAGE batch only when an
independent probe sees boundary switching.

## Ownership boundary

The following components remain inherited from the architecture and formulas
attributed to Đức's proposal:

- exact feasible branch-wise forward;
- safe branch certificate/pruning, Eq. (11)-(12);
- compact target `q`, `beta=2`, and fixed `rho` within the experiment;
- full local plus switching gradient, Eq. (17)-(18);
- optimizer-derived switching normals, Eq. (19)-(21);
- exact ordinary `q/p` weighting for the base/local/focused multiscale
  proposal;
- no differentiation through certificate, proposal construction or solver.

The following Phase 15 components are new work in this repository and must not
be attributed to the original proposal without separate evidence:

1. a controller that selects the stochastic SAGE sample budget;
2. an independent four-sample switching probe;
3. the discrete `N in {4,16,32}` budget policy;
4. controller-accounting fields and the matched training benchmark below.

The controller changes computation, not the declared smoothing radius or
target density.

## Frozen controller

At every training center:

1. solve the exact center and construct the radius-`rho` switching
   certificate;
2. if the full ball is certified stable, allocate the minimum `N=4` SAGE
   batch; the existing certificate removes unnecessary global searches while
   the sampled fixed-branch gradients continue to target the declared
   smoothed objective;
3. otherwise draw four antithetic compact-`q` probe samples and solve their
   exact forwards;
4. count probe winners different from the center winner and select:
   - zero switches: `N=4`;
   - one switch: `N=16`;
   - at least two switches: `N=32`;
5. discard the probe for gradient estimation, draw an independent multiscale
   proposal batch of the selected size, and apply the unchanged exact `q/p`
   SAGE estimator.

Conditional on the certificate/probe and selected `N`, the estimation batch is
independent and uses the same valid estimator. This avoids the optional-
stopping bias that would arise if the probe gradients were reused in the final
average. The probe still costs exact searches and is included in every compute
table. In particular, the stable path does not substitute the center
hard-local gradient: that would silently change the declared smoothed target.

## Frozen pilot

The primary pilot is the Phase 13 3x4 native-disjunction Job-Shop family:

- five fixed seeds;
- 16 train, 12 IID and 12 OOD instances per seed;
- eight heuristic-pretraining epochs;
- 12 matched updates from identical pretrained weights;
- fixed `rho=0.5`, learning rate `1e-3`;
- a shared eight-probe pretrained OOD switching partition;
- methods: hard-local, STE, fixed multiscale SAGE `N=4`, fixed multiscale SAGE
  `N=32`, and adaptive SAGE.

No method selects a checkpoint, radius, seed or evaluation stratum from its
own result. Method execution order is rotated by seed. Hard-local and STE are
reported as direct controls; the registered controller comparison is against
the two fixed SAGE budgets because Phase 15 addresses SAGE's own variance/work
trade-off.

## Frozen gates

`PHASE15_ADAPTIVE_CONTROLLER=PASS` requires:

1. all 25 runs complete with exact IID/OOD feasibility, no unresolved search,
   no logical failure and maximum violation at most `1e-6`;
2. every adaptive step is classified as certified-stable or probed-nonstable
   and executes exactly one independent SAGE estimation batch;
3. every nonstable adaptive call uses exactly four probe searches, every call
   chooses only `N=4,16,32`, and recorded estimation samples equal the selected
   budgets;
4. adaptive mean OOD task loss is no more than `1.10` times the better of the
   two fixed-SAGE mean losses;
5. adaptive perturbed global searches, including probes, do not exceed fixed
   SAGE `N=32` searches.

The joint quality-and-compute claim passes only if both items 4 and 5 pass.
The method is not required to beat STE or hard-local in this phase; those
comparisons and paired 95% intervals remain mandatory outputs. The number of
times each budget is selected is also an outcome, not a tunable gate. A
negative pilot remains versioned and the thresholds will not be changed after
the first full run.

## Smoke implementation check (before pilot)

The one-seed, two-update smoke profile completed all safety, controller-
accounting, quality and compute checks. Neither update received a full-ball
stability certificate. The independent probes observed two switches across
eight searches and selected `N=16` twice. Adaptive SAGE used 26 perturbed
global searches versus fixed-`N=32`'s 32, with OOD losses `0.19317` and
`0.19092`, respectively. This is an implementation check only; no frozen
policy, pilot configuration or gate was changed after observing it.

## Frozen pilot result

The unchanged five-seed pilot completed safely but reported
`PHASE15_ADAPTIVE_CONTROLLER=FAIL` because the pre-registered compute and joint
utility gates failed. Completion, exact safety, controller accounting and
quality gates passed.

| method | OOD loss | higher-switch loss | mean seconds | global searches | branch solves | mean ESS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard-local | 0.13398 | 0.13023 | 3.49 | 0 | 154 | — |
| STE | 0.12693 | 0.12389 | 3.71 | 0 | 96 | — |
| fixed SAGE N=4 | 0.13457 | 0.13216 | 4.94 | 135 | 1230 | 3.00 |
| fixed SAGE N=32 | 0.12867 | 0.12595 | 14.54 | 1120 | 4906 | 25.35 |
| adaptive SAGE | 0.13037 | 0.12692 | 14.81 | 1212 | 4665 | 20.50 |

Adaptive SAGE was within `1.32%` of the better fixed-SAGE mean loss, satisfying
the frozen 10% quality margin, and improved over fixed `N=4`. Its paired OOD
loss difference from fixed `N=32` was `+0.00170` with 95% CI
`[-0.00105, 0.00445]`, so no quality difference is established. It used 4.9%
fewer complete branch solves than fixed `N=32`, but `8.21%` more perturbed
global searches and slightly more wall time. Therefore the registered compute
claim is false.

No one of the 60 training centers obtained a full-ball stability certificate.
All were probed; 240 probes observed 126 switches. The controller chose
`N=4/16/32` on `6/12/42` updates. Because 70% of updates selected the maximum
budget, the independent four-search probe was usually overhead on top of a
near-`N=32` computation. The common pretrained OOD audit independently showed
mean lower/higher switching exposure `0.117/0.496`, consistent with this being
a genuinely boundary-active regime rather than an accounting error.

Against the cheap controls, adaptive minus STE mean OOD loss was `+0.00344`
with 95% CI `[-0.00043, 0.00731]`; adaptive minus hard-local was `-0.00361`
with CI `[-0.00888, 0.00166]`. Neither comparison is statistically resolved,
while STE remains the best mean and cheapest method.

## Scientific conclusion

Phase 15 does not improve the accepted SAGE method. It establishes a useful
negative design constraint: an exact independent-probe controller preserves
the estimator contract, but its probe cost is not amortized when boundary
exposure is high and full-ball certificates are rare. The result must not be
presented as an adaptive-SAGE success.

A subsequent controller should avoid new exact probe solves. Candidate routes
include a conservative budget derived from certificate margins/competitive
groups already computed by SAGE, or a hard per-update solver budget with
cross-step state. Either is new work beyond Đức's proposal and requires a new
pre-registration; Phase 15 thresholds will remain frozen.

## Packaging verification

- the targeted policy/accounting tests passed `16/16`;
- the complete package regression suite passed `3553/3553` in `2m31.7s`;
- the Phase 10-14 multiscale path is shared through one helper, and the full
  suite confirms that the refactor did not regress earlier gates;
- the Kaggle notebook defaults to standalone `RUN_MODE = "phase15"` and
  explicitly expects the frozen `PHASE15_ADAPTIVE_CONTROLLER=FAIL` token,
  recording `NEGATIVE_OVERHEAD_REPRODUCED` rather than a false success;
- generated Phase 15 result directories are ignored, while the full aggregate
  table, intervals, controller counts and scientific conclusion remain in this
  versioned log.
