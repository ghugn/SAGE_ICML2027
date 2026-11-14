# Phase 10 - Native-disjunction job-shop benchmark

Date: 2026-11-14

Status: protocol pre-registration before implementation and before observing
any Phase 10 training comparison.

## Research question

Does SAGE's proposal-aligned switching correction become useful when the
disjunction is intrinsic to the task and branch changes correspond to machine
order decisions, rather than class boxes added to an easy regression dataset?

The goal is not to construct a dataset on which SAGE must win. The benchmark
must retain hard-local and STE even if they are stronger. A negative result is
an estimator diagnosis, not grounds for changing the task after observation.

## Why Iris is not extended

Larger tabular classification data would add samples and classes but would not
make the logical constraints native. Job-shop scheduling has the exact linear
disjunction required by the current backend. For operations `a` and `b` on the
same machine, feasibility requires

```text
start[a] + duration[a] <= start[b]
OR
start[b] + duration[b] <= start[a].
```

Within-job precedence is globally linear. This separates the scientific
question from dataset size: switching a branch changes a real machine order.

Primary benchmark references:

- OR-Library job-shop instances and Taillard-style generator:
  https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html
- official OR-Tools formulation of precedence and no-overlap constraints:
  https://developers.google.com/optimization/scheduling/job_shop

## Controlled Phase 10 scope

The first controlled family uses three jobs and three machines:

- nine normalized processing-time/route operations;
- nine pairwise machine-order disjunctions and at most 512 complete branches;
- input: durations plus one-hot machine routes;
- output: nine normalized operation start times;
- deterministic train/IID/OOD generation;
- a JuMP/HiGHS MILP oracle minimizes makespan, then canonically breaks ties by
  minimizing weighted start times;
- a deterministic serial-dispatch schedule supplies a realistic but generally
  suboptimal pretrained initialization;
- exact lazy branch search supplies every constrained forward.

The OOD slice widens processing-time variation and changes the route seed. It
does not change constraints after seeing method performance.

## Proposal contract retained

1. Eq. (7)-(10): exact feasible branch-wise forward.
2. Eq. (11)-(12): stability certificate and safe competitor pruning.
3. Eq. (17)-(18): fixed-reference local contribution plus switching residual.
4. Eq. (19)-(21): geometry proposal built from optimizer-derived switching
   normals with exact `q/p` importance weights.
5. Black-box and SAGE-Cert receive matched compact-`q`, `rho`, `N` and common
   antithetic samples. Geometry SAGE receives the same target `q`, `rho`, `N`
   and reports ESS.

## Methods and metrics fixed before implementation

Methods:

- pretrained raw and pretrained exact;
- hard-local implicit gradient;
- straight-through estimator;
- black-box stochastic smoothing;
- SAGE-Cert;
- SAGE-Geometry.

Primary task metrics:

- exact schedule feasibility;
- normalized makespan gap to the MILP oracle;
- task loss and start-time MSE;
- machine-order agreement with the canonical oracle;
- branch-switch success from the serial-dispatch initialization;
- perturbed global searches, complete branch solves, wall time, certificate
  rate, switching-sample rate and geometry ESS.

## Hard requirements fixed before smoke

1. Every oracle label and every constrained output must pass the original
   precedence/no-overlap validator. Unresolved outputs fail safety.
2. The MILP oracle must report an optimal schedule for every retained instance.
3. All methods begin from identical pretrained weights and use identical
   per-seed sample order.
4. Data generation and split seeds are deterministic; normalization is fit on
   training data only.
5. The benchmark must show nontrivial initial order disagreement and actual
   perturbation-induced switching; otherwise it is not a discontinuity test.
6. Quality comparisons use both matched updates and explicit solver-work/time
   reporting. Reduction in global searches must not be described as reduction
   in every solve category.
7. Smoke runs may select a feasible runtime budget only. Pilot quality gates
   will be frozen after smoke and before the multi-seed pilot is observed.

## Planned implementation entries

Entries will be appended chronologically. Failed smoke/pilot runs and all gate
changes will be retained.

### 2026-11-14 - implementation and first retained smoke failure

Implemented `phase10_jobshop_common.jl`, the MILP oracle, deterministic
3-by-3 generator, serial-dispatch pretraining, exact schedule evaluation and
five matched backward methods. Targeted tests passed `45/45` before smoke.

The first smoke used one seed, 8/6/6 train/IID/OOD instances, three updates,
`N=2` and `rho=0.10`. Exact feasibility and the native adaptation condition
passed, but the benchmark correctly failed as a discontinuity test:

- black-box switching rate: `0.0`;
- certificate/geometry global-search reduction: `0.0 / 0.0`;
- learning gate: `FAIL`;
- switching and compute gates: `FAIL`;
- acceptance token remained diagnostic-only:
  `PHASE10_JOBSHOP_SMOKE=COMPLETE`.

The smoke also exposed a data-pipeline defect. With only eight train instances,
some one-hot route coordinates were constant. Dividing later test values by a
`1e-6` scale floor produced raw-network outputs around `1e5`, so the
unconstrained base loss was meaningless. The floor will be changed to `0.1`,
which is a representation repair independent of method performance.

Before another training comparison, a radius sweep will select the smallest
fixed `rho` with nonzero switching coverage. This selection uses only branch
switch counts, not task loss or which method wins. The selected radius will be
frozen before the multi-seed pilot.

### 2026-11-14 - switching-only radius selection

After repairing train-only normalization, a deterministic sweep measured
branch changes over 64 antithetic perturbations without updating any method:

| rho | switching samples | rate |
|---:|---:|---:|
| 0.10 | 0/64 | 0.0000 |
| 0.25 | 0/64 | 0.0000 |
| 0.50 | 13/64 | 0.2031 |
| 0.75 | 20/64 | 0.3125 |

`rho=0.50` is the smallest tested radius with nontrivial coverage and is fixed
for the second smoke and proposed pilot. This choice was made before observing
any method quality at that radius.

### 2026-11-14 - second smoke and frozen pilot gates

The repaired smoke at `rho=0.50` produced a genuine discontinuity comparison:

- black-box switching rate: `2/6 = 33.3%`;
- pretrained-exact OOD task loss: `0.08418`;
- hard-local/STE/black-box OOD task loss:
  `0.08193 / 0.07861 / 0.10142`;
- SAGE-Cert/SAGE-Geometry OOD task loss:
  `0.08534 / 0.06613`;
- all constrained IID/OOD CSAT: `1.0`;
- geometry ESS: `1.978/2`;
- safety, adaptation, switching, learning, matched-quality and ESS gates:
  `PASS`;
- global-search reduction: `0%`, so the compute gate remained `FAIL`;
- `PHASE10_JOBSHOP_SMOKE=COMPLETE`.

The result is encouraging but is one seed with three updates and is not a
claim. It also shows that the hard slice removes the certificate speed path:
SAGE-Geometry used 1,431 complete branch solves versus black-box's 846 and was
slightly slower. Phase 10 therefore tests proposal efficiency as **quality or
computation**, consistent with the earlier proposal contract, while retaining
the standalone compute failure.

The multi-seed pilot is frozen before observation:

- five paired seeds;
- 16 train, 12 IID and 12 OOD instances per seed;
- eight serial-dispatch pretraining epochs;
- six fine-tuning updates, `N=4`, `rho=0.50`, learning rate `1e-3`;
- exact-feasibility and zero-unresolved gate;
- mean serial-dispatch gap at least `10%` and pretrained exact order agreement
  below `95%`;
- black-box switching rate at least `2%`;
- SAGE-Geometry improves its pretrained exact OOD task loss;
- SAGE-Cert and SAGE-Geometry each remain within `1.20x` black-box OOD task
  loss;
- proposal efficiency passes if SAGE-Geometry improves black-box OOD task loss
  by at least `10%` **or** the standalone compute gate passes;
- geometry ESS at least `0.60N`.

SAGE versus STE and hard-local is reported with paired confidence intervals
but is not an acceptance gate. No threshold above will be relaxed after seeing
the five-seed pilot.

### 2026-11-14 - retained pilot abort and branch-QP repair

The first execution of the frozen pilot was intentionally aborted rather than
silently dropping a sample. Seed 1 completed hard-local, STE, black-box and
SAGE-Cert, then SAGE-Geometry failed at update 5 on training instance 7:

- estimator status: `uncertified`;
- geometry construction status: `ok` with five switching normals;
- unresolved perturbations: `1/4`;
- the lazy and exhaustive searches both isolated the same complete branch;
- branch choices: `[1,1,1,2,2,1,1,1,2]`;
- HiGHS QP status: `OTHER_ERROR`;
- the LP feasibility retry status: `OPTIMAL`.

Therefore this was a numerical backend failure, not unsafe pruning and not a
failure of Eq. (19)-(21). The existing enumerative active-set fallback was
designed for the three-output cooling model and exhausted its 100,000-set
guard in the nine-output scheduling model.

Implemented a primal feasible active-set projection fallback. It begins at
the feasible point certified by the LP retry, iteratively adds blocking
inequalities or removes negative-multiplier working constraints, and returns
only after the original branch and logical validators accept its result. The
old enumerative fallback remains a final small-problem fallback. No seed,
sample, radius, data instance or acceptance gate was changed.

A regression test now fixes the exact nine-dimensional perturbation and
complete branch that caused the abort. The targeted Phase 10 suite passes
`48/48`, including the new high-dimensional QP fallback test. The frozen pilot
will be restarted from seed 1; partial results from the aborted process are not
used in the reported comparison.

### 2026-11-14 - frozen five-seed pilot result

The repaired pilot was restarted from seed 1 and completed all 25 matched
fine-tuning runs. The pre-registered overall diagnostic gate **failed**; no
threshold was changed after observation.

Configuration and adaptation audit:

- seeds/train/IID/OOD: `5/16/12/12`;
- updates/samples/radius: `6/4/0.50`;
- mean/min serial-dispatch makespan gap: `93.17% / 61.29%`;
- black-box switching rate: `40%`;
- maximum oracle violation: `2.43e-8`;
- every constrained IID/OOD output was feasible (`CSAT=1.0`).

Mean OOD task loss:

| method | OOD task loss | OOD makespan gap |
|---|---:|---:|
| pretrained exact | 0.117850 | 0.593818 |
| hard-local | 0.105711 | 0.559440 |
| STE | 0.104546 | 0.554819 |
| black-box | 0.111687 | 0.577379 |
| SAGE-Cert | 0.106903 | 0.565260 |
| SAGE-Geometry | 0.108973 | 0.574251 |

SAGE-Geometry improved the exact pretrained initialization by `7.53%` and
improved black-box by `2.43%`. Its paired difference from black-box was
`-0.002714` with 95% CI `[-0.009422, 0.003994]`, so this pilot does not support
a superiority claim. STE and hard-local were numerically better; their paired
confidence intervals versus SAGE-Geometry also crossed zero.

Compute outcome:

- black-box/SAGE-Cert/SAGE-Geometry perturbed global searches:
  `120/120/120`;
- complete branch solves: `13,993/25,352/25,384`;
- mean training seconds: `17.61/17.81/17.84`;
- SAGE-Geometry mean ESS: `3.959/4`.

Safety, adaptation, switching, learning, matched-quality and ESS gates passed.
Both the standalone compute gate and the frozen quality-or-compute gate failed,
so the acceptance token is `PHASE10_JOBSHOP_DIAGNOSTIC=FAIL`.

Interpretation: the 3-by-3 slice is genuinely discontinuous, but at
`rho=0.50` almost every perturbation leaves the certified local region. SAGE
therefore performs the global search anyway and additionally pays for
reference/competitor branch projections. The result is a useful negative
boundary: increasing combinatorial branch count alone is insufficient. The
next application should combine native disjunctions with sizeable stable
regions (or temporal reuse) so Eq. (11)-(12) can actually remove work. This
result must not be presented as Phase 10 passing or as evidence that SAGE beats
STE/hard-local on job-shop scheduling.

### 2026-11-14 - packaging and regression verification

- Added `phase10` notebook mode and made it the default Kaggle mode.
- The notebook requires the exact recorded token
  `PHASE10_JOBSHOP_DIAGNOSTIC=FAIL` and labels the outcome
  `NEGATIVE_BOUNDARY_REPRODUCED`; it never converts the result to `PASS`.
- Phase 10 is intentionally excluded from notebook `full` mode, which remains
  the suite of accepted gates plus package tests.
- Added CSV/JSON output routing and ignored generated Phase 10 result folders.
- Updated the README with the model, oracle, methods, exact negative result and
  scientific limitation.
- Validated notebook JSON structure locally.
- Full package regression passed `3208/3208` in `1m40.2s` on Julia 1.13.3.

The local pilot artifacts remain untracked under
`phase10_jobshop_results/`. Kaggle writes the same three artifacts into its
timestamped reproduction log bundle.

### 2026-11-14 - post-pilot interpretation and next hypothesis

Phase 10 does **not** establish that the complete SAGE architecture has
failed. It rejects the narrower hypothesis that increasing branch count and
using one fixed smoothing radius is sufficient to create a SAGE advantage on
the controlled 3-by-3 job-shop slice.

Evidence retained for this distinction:

- SAGE-Geometry improved its exact pretrained initialization by `7.53%`;
- SAGE-Geometry improved mean black-box OOD task loss by `2.43%`;
- every constrained output remained logically feasible;
- geometry importance sampling retained mean ESS `3.959/4`;
- the paired superiority intervals crossed zero;
- STE and hard-local were numerically better;
- all `120/120` SAGE perturbations required global search, so the certificate
  path produced no global-search reduction.

The observed failure mode is a fixed-radius trade-off. Small radii in the
switching-only sweep did not reach boundaries, whereas the selected
`rho=0.50` produced useful switching but moved too many samples outside the
certified stable region. One radius was therefore being asked to serve two
different roles: cheap local/certificate estimation and boundary discovery.

The next pre-registered hypothesis should remain aligned with the proposal's
architecture while testing this missing mechanism:

1. keep the Phase 10 fixed-radius result unchanged as the negative baseline;
2. compare fixed radius against margin-adaptive radius, or explicitly separate
   a local certificate radius from a boundary-proposal radius;
3. select radii only from branch-margin/geometry information, never from test
   loss or the winning method;
4. require exact target-density/importance-weight accounting, feasibility and
   zero unsafe pruning;
5. measure switching coverage, global searches, complete branch solves, bias,
   ESS, OOD task loss and paired uncertainty;
6. treat failure to improve both quality or computation across paired seeds as
   evidence against applying certificate-geometry SAGE to this job-shop
   regime.

No adaptive-radius result exists yet. It must not be described as a repair or
advantage until the ablation is implemented and evaluated under gates frozen
before observation.
