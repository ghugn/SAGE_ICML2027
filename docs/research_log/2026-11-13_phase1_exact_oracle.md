# Phase 1 Research Log — Exact Exhaustive Branch Oracle

Date: 2026-11-13  
Runtime: Julia 1.13.3  
Parent baseline: `2026-11-13_phase0_baseline.md`

## Entry 0 — Objective and acceptance gate

### Objective

Build a small-instance ground-truth projector for the original union of convex
branches. This oracle is intentionally exhaustive and is not the scalable
pruning algorithm.

For each complete branch assignment `k`, solve

```text
minimize    0.5 * ||y - yhat||^2
subject to  bounds and global constraints
            all constraints in the selected disjunct of every rule
```

Then select the feasible branch with smallest objective. Ties are resolved
lexicographically by branch choice so results are deterministic.

### Required guarantees

1. Branch combinations are generated lazily rather than materialized as a full
   Cartesian-product array.
2. Every accepted branch output is rechecked against the original Boolean
   model using `check_feasibility`.
3. The global result is marked exact only if every branch is either solved to
   optimality or proved infeasible.
4. Unknown solver statuses or numerically invalid outputs produce an
   `:uncertified` result, never a false exact certificate.
5. Closed-form interval/box cases, infeasible cases, ties, and the Phase 0
   counterexample must pass deterministic tests.

### Default experimental configuration

- Distance: Euclidean, objective `0.5 * squared_distance`.
- Continuous branch solver: HiGHS.
- Logical feasibility tolerance: `1e-6`.
- Objective tie tolerance: `1e-10`.
- No Gurobi or commercial solver is assumed.

## Entry 1 — Oracle implementation

### Files and semantics

- `src/exact/branch_types.jl` defines per-branch and aggregate results with
  explicit `:optimal`, `:infeasible`, and `:uncertified` semantics.
- `src/exact/branch_projection.jl` builds a continuous QP for exactly one
  original branch. It does not use gamma variables or a convex hull.
- `src/exact/exhaustive_oracle.jl` lazily iterates Cartesian choices, collects
  diagnostics, and applies deterministic lexicographic tie-breaking.
- A default `max_branches=100_000` guard prevents accidental exponential runs.

### Certification policy

An `OPTIMAL` solver status is necessary but not sufficient: the returned point
must pass both `check_feasibility` and a residual check of the specifically
selected disjuncts. Checking only the union would be insufficient because a
point could satisfy a different disjunct from the branch being certified. If
any branch returns an unknown status or fails numerical verification, the
aggregate result is `:uncertified`, even if a feasible incumbent exists. This
is required because an unresolved branch could contain a better solution.

### Initial smoke test

For `y <= 2 OR y >= 5` and `yhat=4.2`, the oracle returned:

```text
status=optimal
choices=[2]
y=[5.0]
objective=0.32
resolved=2/2
```

This matches the analytic Euclidean projection.

## Entry 2 — Test design

The new suite covers:

1. analytic 1D interval projections, including a symmetric tie;
2. a symmetric 2D four-branch box case with deterministic tie-breaking;
3. the Phase 0 relaxation counterexample, whose exact solution must be
   `[0.75, 0.1]` on branch `[2, 1]`;
4. partially infeasible and globally infeasible branch sets;
5. an ordinary convex model with zero disjunctions;
6. invalid inputs and the combinatorial safety guard;
7. optional suppression of stored per-branch results for lower memory use.

Additionally, 24 seeded random projections in 2D and 3D are compared against
an independent closed-form coordinatewise oracle for separable two-interval
sets. The seed is `20261113` and the expected projection is computed without an
optimization solver.

## Entry 3 — Scope boundary

The oracle establishes ground truth by exhaustive branch-wise convex QPs. It is
not a monolithic MIQP backend and is not intended for large products. An
independent commercial-MIQP cross-check would be useful when a licensed solver
is available, but the current acceptance suite does not claim that experiment
was performed. Independent validation currently comes from closed-form 1D,
2D, and 3D cases plus original-logic residual certificates.

## Entry 4 — Clean test-environment dependency failure

The first full `Pkg.test()` attempt failed before the oracle tests because
`Random` was available in the development environment but was not declared in
the package test target:

```text
ArgumentError: Package Random not found in current path.
```

This did not affect direct `test/runtests.jl` execution, which is why both test
modes are required. `Random` was added to `[extras]` and the `test` target in
`Project.toml`. The failure is retained here rather than omitted from the
experimental record.

The first edit used an incorrect remembered UUID for `Random`, and the next
`Pkg.test()` correctly rejected it as unregistered. The UUID was then read from
Julia itself via `Base.PkgId(Random).uuid` and corrected to
`9a3f8284-a2c9-5f02-9a11-845980a1fd5c`. This second failure is also retained to
keep the dependency audit complete.

## Entry 5 — Final Phase 1 verification

### Clean package test

After correcting the test target, the full suite passed in a fresh temporary
package environment:

```text
Test Summary:     | Pass  Total   Time
DisjunctiveNet.jl |  375    375  36.5s
Testing DisjunctiveNet tests passed
```

The Phase 1 test file contributes 161 assertions, including 96 assertions from
the seeded random analytic comparison.

### Standalone exact certificate

Command:

```powershell
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. scripts\verify_exact_oracle.jl
```

Observed result for the Phase 0 counterexample:

```text
Total branches: 4
Exact status: optimal
Certified: true
Winning choices: [2, 1]
Exact y: [0.75, 0.09999999000000098]
Exact objective (0.5*d^2): 0.03125000000000005
Exact max logical violation: 0.0
Feasible/infeasible/unresolved branches: (3, 1, 0)
Relaxed DNF logically feasible: false
Relaxed DNF max logical violation: 0.15010814669390604
EXACT_ORACLE_VERIFICATION=PASS
```

### Exit decision

The exhaustive branch oracle is accepted as the internal ground truth for
small controlled instances. Phase 2 may compare lazy search against it using
all of the following hard gates:

- identical certified status;
- identical winning objective within tolerance;
- logically feasible returned point;
- deterministic tie-equivalent branch;
- zero unsafe prunes.

The remaining external-validation item is comparison against an independent
MIQP solver when one is available. This is not a blocker for implementing lazy
search, but it must be completed before making a strong solver-independent
claim in a paper.

