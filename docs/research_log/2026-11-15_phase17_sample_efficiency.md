# Phase 17 — Confirmatory sample-efficiency frontier

Date: 2026-11-15

Status: implementation and systems preflight complete; confirmatory evidence
not yet collected.

## Question

Phase 16 found a long-horizon crossover and, post hoc, made fixed SAGE `N=4`
the most promising compute/quality point. Phase 17 asks a narrower
confirmatory question on new seeds: is `N=4` enough to retain the long-horizon
OOD advantage, and is it non-inferior to `N=32` while using substantially less
solver work?

This phase does not change the SAGE estimator target. It separates a discovery
result (Phase 16 seeds 1-5) from a registered confirmation (seeds 6-10).

## Ownership and attribution boundary

The mathematical core retained from Đức's proposal and the established SAGE
implementation is unchanged:

- exact feasible projection in the forward pass;
- compact target proposal `q`, `beta=2`, and fixed `rho=0.5`;
- certificate and safe competitor pruning from Eq. (11)-(12);
- the smoothed-gradient estimator from Eq. (17)-(18);
- geometry normals and focused proposals from Eq. (19)-(21);
- exact `q/p` importance weights.

Phase 17 additions are implementation-team experimental and systems work, not
claims that they appeared in Đức's original proposal:

- the `N={2,4,8,16,32}` confirmatory frontier;
- new-seed separation and paired statistical gates;
- a platform-calibrated, frozen matched-wall-clock STE control;
- deterministic multi-core preflight and ETA lock;
- exact optimizer-state resume and reload-verified model artifacts.

## Frozen confirmatory protocol

- Problem: the same native-disjunction 3x4 Job-Shop used in Phases 13-16.
- Confirmatory seeds: `6:10`, untouched by smoke, timing calibration and Phase
  16 discovery.
- System-calibration seed: seed 1 from the already observed discovery set.
- Common pretraining: eight epochs; 16 train, 12 IID and 12 OOD instances.
- Fixed-update budget: 1024 optimizer updates.
- SAGE variants: fixed multiscale geometry with `N=2,4,8,16,32`.
- Controls: STE at 1024 fixed updates and STE at a matched-time step count.
- The matched-time STE count is calculated once from the passing preflight as
  `round(1024 * t_step(SAGE-N4) / t_step(STE))`, clamped to `[1024,8192]`,
  then frozen before any seed 6-10 output is observed.
- The method execution order is rotated by seed.
- No best-validation or best-test checkpoint selection is allowed.

Primary registered comparison: paired SAGE `N=4` minus fixed-update STE OOD
task loss at 1024 updates. Superiority requires the upper endpoint of the
five-seed paired 95% interval to be below zero.

Secondary comparisons:

1. SAGE `N=4` versus matched-time STE, with the same superiority rule;
2. SAGE `N=4` versus `N=32`, using a predeclared non-inferiority margin of
   `0.005` OOD task loss;
3. SAGE `N=4` must reduce perturbed global searches by at least 50% relative
   to `N=32` for the combined sample-efficiency gate.

A failed scientific gate remains a valid completed experiment. The runner
prints `PHASE17_CONFIRMATORY=COMPLETE`; individual claims are read from the
summary gates and must not be rewritten as a software failure.

## Mandatory systems preflight

The expensive run is locked unless all conditions pass on the same machine and
source revision:

1. at least two Julia threads are available and at least two are observed;
2. serial and threaded SAGE produce identical final weights and solver work;
3. an interrupted run resumes to the exact uninterrupted final weights and
   work counters;
4. end-to-end threaded speedup for SAGE `N=4` is at least `1.30x`;
5. predicted full confirmatory runtime, after a conservative `1.25x` systems
   margin for serialization, evaluation, pretraining and load drift, is no more
   than four hours.

Every confirmatory method stores an atomic training state every 64 updates.
The state contains network weights, Adam moments and step, training counters,
trajectory, copied requested checkpoints and elapsed time. A final `.jls`
model artifact is deserialized immediately and re-evaluated before the method
is marked complete. Completed method caches are source-revision, thread-count
and hyperparameter signed.

## Local implementation evidence

Commands used (PowerShell environment assignment omitted here for brevity):

```text
julia --project=. scripts/benchmark_phase17_sample_efficiency.jl
julia --threads=4 --project=. scripts/benchmark_phase17_sample_efficiency.jl
```

The smoke profile passed exact safety for STE and SAGE `N=2/4`.

An early 32-update engineering dry run, performed before the seed-separation
audit, was intentionally rejected and is not a valid registered preflight:

- serial/threaded `N=4`: `6.2207 / 4.8240` seconds;
- speedup: `1.2895x`, below the frozen `1.30x` threshold;
- parity: PASS; exact resume: PASS; four sample threads: observed;
- estimated full runtime: `1.85 h`.

No threshold was weakened. After moving calibration to discovery seed 1,
the frozen 64-update window passed. The first registered repetition measured:

- serial/threaded `N=4`: `18.3582 / 11.8866` seconds;
- end-to-end speedup: `1.5444x`;
- parity: PASS; exact resume: PASS; four sample threads: observed;
- local estimated raw training runtime: `2.756 h`; with the frozen `1.25x`
  systems margin the gate estimate is approximately `3.445 h`, below the `4 h`
  ceiling;
- local frozen matched-time STE budget: 2031 updates.

The final post-implementation rerun, including the explicit `1.25x` ETA safety
factor in code, also passed:

- serial/threaded `N=4`: `15.1975 / 10.9669` seconds;
- end-to-end speedup: `1.3858x`;
- parity, exact resume and four-thread-use gates: PASS;
- conservative full-run estimate: `3.256 h`;
- locally frozen matched-time STE budget: 2188 updates.

The variation between the two passing 64-update measurements is why each
Kaggle confirmatory session recalibrates and freezes its own platform plan.

The Kaggle run must repeat the 64-update preflight. Its matched-time count and
ETA are platform-specific and replace the local timing plan; they do not
change the mathematical estimator or fixed-update comparison.

Targeted tests passed `44/44`: uninterrupted checkpoint behavior, exact
training resume, `N=2/4/8` sample-frontier safety, ESS bounds and model
serialization/reload. The Phase 17 smoke token also passed.
The complete package regression suite passed `3609/3609` in `2m33.0s`.

## Files changed

- `scripts/benchmark_phase17_sample_efficiency.jl`: smoke, preflight and
  locked confirmatory runner.
- `scripts/phase10_jobshop_common.jl`: exact resumable training state.
- `test/test_phase16_long_horizon.jl`: uninterrupted-versus-resumed equality.
- `test/test_phase17_sample_efficiency.jl`: frontier and artifact tests.
- `SAGE_Kaggle_Reproducibility.ipynb`: safe default preflight and explicit
  confirmatory unlock.
- `.cursor/rules/benchmark-system-engineering.mdc`: governing benchmark policy
  already added before this phase.

## Limitations and paper-use constraints

- Five paired seeds give only a small-sample interval; the raw per-seed table
  must accompany every aggregate claim.
- Matched-wall-clock results are hardware-specific. The fixed-update result is
  the portable primary comparison.
- Julia `Serialization` artifacts are intended for exact same-revision resume,
  not long-term cross-version model interchange.
- The ETA is an extrapolation from the discovery calibration seed and can drift
  with Kaggle load. The 64-update heartbeat/checkpoint bounds lost work but
  cannot eliminate platform variance.
- The benchmark is still controlled synthetic Job-Shop. A positive Phase 17
  result justifies external benchmark validation; it does not by itself prove
  general superiority.
- No Phase 17 confirmatory quality result exists at the time of this note.

## Kaggle execution

Keep the notebook default `RUN_MODE = 'phase17_preflight'` for the short gate.
Review the printed speedup, ETA and frozen matched-time STE steps. Only then set
`RUN_MODE = 'phase17_confirmatory'`. That mode repeats the preflight on the
current Kaggle machine and source revision, and unlocks the full run only if it
passes. Keep `PHASE17_RUN_NAME` unchanged to resume after interruption.
