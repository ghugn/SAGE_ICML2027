# Phase 16 follow-up — Parallel SAGE sample backend

Date: 2026-11-15

## Motivation

The frozen Phase 16 CPU run exposed sustained utilization near one core even
though Kaggle provisioned four CPU cores. `JULIA_NUM_THREADS=4` only creates a
thread pool; the estimator's perturbation loop remained serial. This made the
five-seed evidence run take roughly a working day and is an implementation
scalability defect, not a property required by the SAGE equations.

This change is post-proposal engineering. It does not alter the exact feasible
forward, compact target distribution, Eq. (17)-(18) estimator, Eq. (19)-(21)
geometry, certificate/pruning semantics or adaptive sample policy. It must not
be claimed as new estimator mathematics.

## Design

Each SAGE perturbation is evaluated independently after the center projection,
stability certificate, geometry and sample batch are fixed. The implementation
now optionally:

1. allocates one result slot per sample;
2. evaluates slots with `Threads.@threads :static`;
3. creates fresh JuMP/HiGHS models inside every sample solve;
4. reduces gradients, records and solver counters in original sample order.

The ordered reduction is required for deterministic floating-point summation
and record ordering. No mutable training counter is shared by worker threads.
The shared disjunctive model is read-only during sample evaluation; all JuMP
models being optimized are newly constructed within the corresponding worker.

The public opt-in is `SAGEConfig.parallel_samples`; the Job-Shop runner exposes
`parallel_sage_samples`, and Phase 16 uses
`SAGE_PHASE16_PARALLEL_SAMPLES=1`. Serial execution remains the library default
until the parallel gate has been reproduced across supported environments.

## Verification contract

Run with at least two Julia threads:

```bash
JULIA_NUM_THREADS=4 julia --project=. scripts/verify_sage_parallel.jl
```

The verification uses matched fixed samples and requires:

- exact serial/threaded gradient equality under ordered reduction;
- identical global-search, branch-solve, proposal-component, feasibility and
  unresolved accounting;
- observation of at least two Julia sample threads;
- lower median wall time across three warmed 32-sample Job-Shop estimates.

The Kaggle notebook runs this gate before its resume smoke test and before any
multi-hour Phase 16 experiment. It will not begin the full experiment unless
`SAGE_PARALLEL_VERIFICATION=PASS` is emitted.

## Files changed

- `src/gradients/sage_estimator.jl`: independent sample work item, deterministic
  threaded execution and thread-use diagnostics.
- `scripts/phase10_jobshop_common.jl`: training/diagnostic opt-in and accounting.
- `scripts/verify_sage_parallel.jl`: parity, utilization and timing gate.
- `scripts/benchmark_phase16_long_horizon.jl`: environment switch, cache
  signature bump and heartbeat thread diagnostics.
- `SAGE_Kaggle_Reproducibility.ipynb`: parallel verification before long runs.
- `test/test_gradient_fidelity.jl`: deterministic serial/threaded estimator
  regression.

## Current verification status

Runtime verification was completed locally with Julia 1.13.3 and four Julia
threads. The deterministic Job-Shop gate observed all four sample threads,
exact serial/threaded gradient and accounting parity, and these warmed timings:

- serial seconds: `[0.3370987, 0.2355616, 0.2502861]`;
- threaded seconds: `[0.2032986, 0.1362496, 0.1329465]`;
- median speedup: `1.83697x`;
- token: `SAGE_PARALLEL_VERIFICATION=PASS`.

The complete regression suite also passed: `3576/3576` tests in `2m01.9s`.
Notebook JSON validation and `git diff --check` passed.

The Phase 16 integration smoke also completed with
`PHASE16_LONG_HORIZON_SMOKE=COMPLETE`. Its SAGE N=4, SAGE N=32 and adaptive
paths each reported `sample_threads=4`; STE and hard-local correctly reported
zero because they do not call the SAGE perturbation estimator.

An attempted implementation that set the HiGHS `threads=1` option on every new
model was rejected during development: it caused `OTHER_ERROR` results in the
existing exact-oracle suite on Windows. That option was removed before the
accepted measurements above. The accepted design parallelizes independent
Julia work items without mutating solver-global thread configuration.

## Limitations

- only the per-sample SAGE solve is threaded; center search, geometry,
  evaluation, neural updates and adaptive probing remain serial;
- small `N` may become slower because thread scheduling has fixed overhead;
- custom solver constructors and user-provided loss callbacks must themselves
  be safe for independent concurrent calls;
- speedup is bounded by serial work, memory bandwidth and solver variability;
- the already-running serial Phase 16 experiment remains the valid baseline
  and is not mixed with results from this backend.
