# Phase 16 — Long-horizon convergence study

Date: 2026-11-14

## Motivation

Phases 10-15 are mechanism and short-horizon adaptation studies. In the 3x4
Job-Shop experiments, 12 updates with 16 training instances are only 0.75
exact-training epochs; the longest Phase 13 cell used 24 updates, or 1.5
epochs. Eight heuristic-pretraining epochs do not replace training with each
candidate backward method. Therefore the existing ranking cannot be presented
as an asymptotic or converged-training result.

Phase 16 asks whether the SAGE/STE/hard-local ranking changes after materially
longer exact training.

## Ownership boundary

The five backward methods are frozen from earlier phases:

- hard-local and STE are external controls;
- fixed multiscale SAGE `N=4` and `N=32` retain the exact feasible forward,
  Eq. (11)-(12) certificate/pruning, Eq. (17)-(18) gradient, Eq. (19)-(21)
  geometry and exact `q/p` attributed to Đức's proposal and its established
  implementation;
- adaptive SAGE is the new Phase 15 independent-probe controller and remains
  frozen despite its failed compute gate.

Phase 16 adds only experimental infrastructure: long-horizon checkpointing,
work-at-checkpoint accounting and convergence classification. These additions
are not part of Đức's original proposal and do not modify any backward.

## Frozen pilot

- native-disjunction 3x4 Job-Shop;
- five fixed seeds;
- 16 train, 12 IID and 12 OOD instances per seed;
- identical eight-epoch heuristic pretraining;
- fixed `rho=0.5`, learning rate `1e-3`, no learning-rate tuning;
- checkpoints at `16, 64, 256, 1024` optimizer updates, corresponding to
  `1, 4, 16, 64` exact-training epochs;
- methods: hard-local, STE, fixed SAGE `N=4`, fixed SAGE `N=32`, and the
  unchanged Phase 15 adaptive controller;
- one uninterrupted run per seed/method; checkpoint networks are copied from
  that run, so no horizon is independently restarted or selected;
- a shared eight-probe switching-exposure partition built only from the common
  pretrained network.

The primary comparison is matched optimizer updates. Elapsed CPU time and
solver work are reported at every checkpoint, but Phase 16 does not grant cheap
methods extra updates. Earlier phases already establish their wall-time
advantage; the current question is specifically convergence under sufficient
epochs.

No early stopping or best-test checkpoint is used for the primary table. All
checkpoints are reported. The final 1024-step result is primary.

## Frozen classification

At 1024 steps, compute paired five-seed OOD-loss differences for fixed SAGE
`N=32` minus STE and hard-local. Report:

- `OBSERVED` only if fixed `N=32` has lower mean OOD loss than both controls
  and at least one corresponding paired 95% interval has upper bound below
  zero;
- otherwise `NOT_OBSERVED`.

This deliberately strict classification tests the hypothesis raised after
Phase 15: SAGE may require longer training before its more faithful gradient
becomes useful. Adaptive SAGE and `N=4` remain secondary curves and cannot
replace the registered fixed-`N=32` comparison.

## Frozen completion gates

`PHASE16_LONG_HORIZON=PASS` requires only:

1. all `5 seeds x 5 methods x 4 horizons = 100` checkpoint rows are emitted;
2. every checkpoint network is exact-feasible on IID/OOD and both switching
   strata, with no unresolved result, no logical failure and maximum violation
   at most `1e-6`;
3. steps, elapsed time, global searches and branch solves are finite and
   nondecreasing within every uninterrupted seed/method run;
4. every SAGE checkpoint has positive finite ESS after training begins;
5. adaptive controller accounting remains exact at every checkpoint;
6. the final checkpoint is exactly 1024 updates for every run.

The crossover classification is not a completion gate. A negative answer is a
valid long-horizon scientific result and will not be converted into a software
failure.

## Runtime policy

The complete pilot is intentionally not run on the development machine. Based
on Phase 15 timings, five seeds and two high-budget SAGE variants can require
several CPU hours, and Kaggle may be slower. Local execution is limited to a
small smoke profile. The committed Kaggle notebook is the designated pilot
runner and will save CSV/JSON logs at every completed stage.

## Smoke implementation check (before Kaggle pilot)

The one-seed smoke profile used checkpoints at two and four updates and passed
completion, exact safety, monotone-work, SAGE-ESS, adaptive-accounting and
final-checkpoint gates. Networks were copied from one uninterrupted four-step
run per method. At the four-step checkpoint, fixed SAGE `N=32` had OOD loss
`0.18641`, versus STE `0.18867` and hard-local `0.19380`; this single-seed
smoke observation is not scientific crossover evidence and did not change the
frozen pilot or classification.

The smoke also exposed why elapsed time must be stored at checkpoints rather
than inferred from final runtime: evaluation work between checkpoints is part
of the actual runner wall clock. Solver searches and branch solves were
monotone for every method, and all exact outputs remained feasible.

## Packaging verification

- the targeted uninterrupted-checkpoint regression passed `10/10`;
- the full package suite passed `3563/3563` in `2m10.4s`;
- the local smoke runner passed every frozen software gate;
- after every completed seed/method run, the runner rewrites the
  accumulated checkpoint/trajectory/audit CSVs and `phase16_progress.json`, so
  completed method results survive a later-stage interruption;
- the Kaggle notebook defaults to `RUN_MODE = "phase16"`, requires the neutral
  completion token and records either crossover outcome without treating it as
  a software failure;
- generated Phase 16 output directories are ignored, while this pre-
  registration and the eventual Kaggle CSV/JSON bundle preserve the paper
  audit trail;
- the 1024-step five-seed pilot was deliberately not executed locally.

## 2026-11-15 restart hardening

An interrupted long CPU run exposed an operational gap: accumulated CSVs
survived after each completed method, but restarting the runner did not consume
them and the timestamped notebook directory prevented automatic continuation.
This is an evaluation-infrastructure defect, not a change to the SAGE
estimator, certificate, proposal distribution or adaptive policy.

The runner now:

- writes one atomic serialized cache after every completed `seed-method` pair;
- validates the profile, seed, method, horizon, sample budget, radius and
  learning rate before loading a cached result;
- reconstructs the accumulated CSV/JSON evidence from validated caches on
  restart and skips the corresponding expensive training calls;
- prints a flushed heartbeat at every recorded checkpoint/evaluation interval
  (64 updates in the frozen pilot), including elapsed time and global searches;
- uses a stable, user-visible `PHASE16_RUN_NAME` in the Kaggle notebook so a
  kernel restart targets the same output directory.

The recovery boundary is deliberately conservative: the currently executing
method is repeated after interruption, while every fully completed method is
reused. The frozen five-method, five-seed, 1024-update scientific design and
all acceptance criteria remain unchanged. This hardening is evaluation-only
and must not be described as a methodological contribution.
