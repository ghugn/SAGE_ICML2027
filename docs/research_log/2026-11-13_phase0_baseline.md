# Phase 0 Research Log — Baseline Stabilization and Output Verification

Date: 2026-11-13  
Runtime: Julia 1.13.3  
Project: DisjunctiveNet.jl

This file is the audit trail for Phase 0. Every material code change records the
hypothesis, implementation, verification command, and result so the experiment
can later be reconstructed for a paper.

## Entry 0 — Reproducible starting point

### Objective

Establish an honest baseline before implementing adaptive branch search or SAGE.
"Stable" means:

1. the complete test suite is reproducible;
2. the forward and backward passes use the same formulation configuration;
3. degenerate KKT systems do not crash training;
4. output is checked against the original disjunctive logic, separately from
   feasibility in a convex-hull relaxation.

### Environment

- Official portable Julia 1.13.3 is the primary runtime.
- The project-local depot is `../.julia_depot`.
- Registered DiffOpt 0.6.2 is used. The repository-recommended DiffOpt fork was
  rejected for this baseline because it does not parse on Julia 1.13.3.

### Starting measurement

Command:

```powershell
$env:JULIA_DEPOT_PATH=(Resolve-Path '..\.julia_depot').Path
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. -e 'using Pkg; Pkg.test(; coverage=false)'
```

Observed result:

- 171 passed;
- 0 assertion failures;
- 5 errors;
- all 5 errors were `SingularException` raised by DiffOpt while solving the
  reverse-mode KKT system.

The basic example solved successfully, but returned approximately
`[0.59989, 0.20011]` for the rule `y1 <= 0.25 OR y1 >= 0.75`. This point violates
the original rule and is only feasible in its convex-hull relaxation. Therefore
solver status `OPTIMAL` alone is not a valid logical-output certificate.

### Audit findings to fix

1. The layer's custom reverse rule omitted `num_dnf_rules` and `rule_ordering`.
2. DiffOpt's default quadratic-program linear solver performs a direct solve and
   throws on singular KKT matrices.
3. No public utility distinguishes original logical feasibility from relaxation
   feasibility.
4. `mode=:milp` is accepted by configuration even though the call path still
   executes the convex-hull QP. This option must not silently claim MILP
   semantics.

## Change entries

Subsequent entries are appended as each change is implemented and verified.

## Entry 1 — Original-logic output certificate

### Hypothesis

The existing `OPTIMAL` status certifies only the solved convex-hull formulation.
It cannot certify membership in the original nonconvex feasible set.

### Change

- Added `FeasibilityReport`.
- Added `constraint_violation`, `check_feasibility`, and
  `is_logically_feasible`.
- The checker evaluates bounds, global constraints, and every original OR rule.
  It records the best disjunct and residual for each rule.
- Added tests including an explicit optimal-but-logically-infeasible relaxation
  output.

### Paper relevance

All later pruning experiments must report both solver status and original-logic
violation. This prevents a stronger runtime number from being obtained by
silently measuring a relaxation rather than the intended feasible set.

## Entry 2 — Honest configuration semantics

### Hypothesis

Configuration labels are unsafe if they do not alter execution.

### Change

- `mode=:milp` is rejected until an actual MILP projection backend exists.
- `gradient=:straight_through` now executes the identity pullback.
- The layer pullback now forwards `num_dnf_rules` and `rule_ordering`, making its
  backward formulation match its forward formulation.

### Paper relevance

Experiments can now state which formulation and gradient estimator actually ran;
the label is no longer decoupled from code execution.

## Entry 3 — Degenerate KKT fallback

### Hypothesis

The five baseline errors are caused by rank-deficient active-set KKT matrices,
not by failed forward optimization.

### Change

- Kept DiffOpt's direct linear solve as the primary path.
- On `SingularException`, install a project-owned `RobustKKTSolver` and retry
  with LSQR, which returns a deterministic minimum-norm least-squares solution.
- No installed package or depot source is modified.

### Risk to track

At a degenerate active set the classical derivative is not unique. The LSQR
result is a stable minimum-norm generalized sensitivity; later experiments must
not describe it as a unique mathematical derivative at switching points.

## Entry 4 — Verification protocol

### Commands

Full regression suite:

```powershell
$env:JULIA_DEPOT_PATH=(Resolve-Path '..\.julia_depot').Path
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. test\runtests.jl
```

Human-readable baseline certificate:

```powershell
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. scripts\verify_baseline.jl
```

The verification script reports, for every formulation, solver status,
objective, original-logic feasibility, maximum violation, and output vector. It
also differentiates through an intentionally rank-deficient KKT system.

### Gradient check

A non-switching affine-equality projection is compared with central finite
differences using step `1e-5`. This is deliberately separate from the
degenerate stress cases: finite differences verify ordinary local sensitivity,
while stress tests verify that the documented minimum-norm fallback is finite.

## Entry 5 — Final Phase 0 result

### Pinned test matrix

- Julia 1.13.3
- DiffOpt 0.6.2
- Flux 0.16.15
- HiGHS 1.26.1
- IterativeSolvers 0.9.4
- JuMP 1.32.1
- MathOptInterface 1.54.0
- Zygote 0.7.18

The complete package test was executed in a fresh temporary test environment:

```text
Test Summary:     | Pass  Total   Time
DisjunctiveNet.jl |  214    214  39.7s
Testing DisjunctiveNet tests passed
```

The standalone certificate ended with:

```text
dnf         | OPTIMAL | logical_feasible=false | max_violation=0.1501081467
cnf         | OPTIMAL | logical_feasible=false | max_violation=0.1500000495
partial_dnf | OPTIMAL | logical_feasible=false | max_violation=0.1500000495
Degenerate KKT gradient: [-1.590696e-10, -1.069769e-10]
BASELINE_VERIFICATION=PASS
```

The `logical_feasible=false` rows are expected and important: Phase 0 verifies
and reports the semantics of the existing convexified baseline. Producing an
exact point in the original union is Phase 1's branch-oracle responsibility.

### Files changed

- `src/core/feasibility.jl`: original-logic feasibility certificate.
- `src/backend/differentiation.jl`: robust KKT retry, honest straight-through
  rule, and forward/backward configuration parity.
- `src/core/utils.jl`: reject the unimplemented MILP mode.
- `src/display/display.jl`: benchmark objective and logical residual reporting.
- `test/test_feasibility.jl`: logical-output regression tests.
- `test/test_differentiation.jl`: smooth finite-difference verification; the
  existing stress and integration tests now exercise the robust reverse path.
- `scripts/verify_baseline.jl`: one-command human-readable certificate.
- examples and README: corrected relaxation semantics and visible diagnostics.

### Exit decision

Phase 0 is accepted as a reproducible and semantically honest convexified
baseline. Phase 1 may now add an exhaustive exact branch oracle without using
the current convex-hull output as a false ground-truth logical projection.

## Entry 6 — Failure-path audit

### Finding

The old layer default `fallback=:identity` could return the unconstrained neural
output after a failed solve. That behavior is impossible to distinguish from a
successful projection when only the returned vector is observed.

### Change

- Changed the default to `fallback=:error` in `ProjectionConfig`,
  `DisjunctiveProjectionLayer`, and `ConstrainedFluxModel`.
- Kept `fallback=:identity` as an explicit opt-in for backward compatibility.
- Added an infeasible-model regression test proving that the default throws and
  the opt-in identity path returns the input.

### Paper relevance

Failed solves can no longer enter evaluation tables as apparent successful
projections. Any identity fallback used by a later experiment must be an
explicit recorded configuration choice.
