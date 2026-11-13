# Pre-SAGE Foundation Gate

Date: 2026-11-13

## Why this gate was added

Phase 4 prototype diagnostics were initially run before one combined command
had re-verified the three foundations in their dependency order. Those Phase 4
numbers are therefore treated as implementation smoke tests, not accepted
research output, until this gate and the complete regression pass.

The three foundations have different provenance and claims:

1. **Baseline stabilization** is engineering around the inherited
   DisjunctiveNet convexified implementation. It is not a SAGE mathematical
   contribution and is not an exact projector.
2. **Original-logic verification** implements membership checking for the
   intersection-of-unions feasible set. It detects invalid relaxation outputs
   but does not prove projection optimality.
3. **Exact ground-truth oracle** directly implements the proposal's branch-wise
   Euclidean projection and exact branch selection, equations (7)-(10), for the
   current fixed-input finite polyhedral scope.

## Hard gate

Run, in order:

```powershell
julia --project=. scripts\verify_foundations.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Phase 4 output may be evaluated only after both commands pass. The foundation
script requires:

- DNF, CNF, and partial-DNF baselines solve reproducibly and report their true
  original-logic status;
- the controlled convex-hull gap is rejected by the logical validator;
- feasible, infeasible-gap, and non-finite points are classified correctly;
- the rank-deficient baseline backward path returns a finite gradient;
- the exact oracle resolves every branch of the controlled counterexample;
- its point, branch, objective, and deterministic tie match analytic values;
- 24 seeded 2D/3D cases match an independent coordinatewise analytic oracle.

## Scope that remains external

This gate does not claim independent commercial MIQP cross-solver validation,
nonlinear convex branches, or input-dependent rule activation. Those remain
explicit paper-facing limitations. Passing this gate certifies the internal
polyhedral foundation used by the Phase 4 prototype; it does not by itself
validate SAGE efficiency or gradient quality.

## Gate execution result

The foundation verifier was executed before the accepted Phase 4 rerun:

```text
[Gate 1] Reproducible convexified baseline
dnf         | OPTIMAL | finite objective | logical_feasible=false
cnf         | OPTIMAL | finite objective | logical_feasible=false
partial_dnf | OPTIMAL | finite objective | logical_feasible=false
finite degenerate-KKT gradient: true

[Gate 2] Original-logic validator
known feasible point accepted: true
logical-gap point rejected: true
non-finite point rejected: true

[Gate 3] Exact branch-wise ground-truth oracle
counterexample winner/status: [2, 1]/optimal
resolved branches: 4/4
deterministic analytic tie winner: [1]
independent analytic random cases: 24/24
FOUNDATION_VERIFICATION=PASS
```

The immediately following isolated package regression reported:

```text
Test Summary:     | Pass  Total     Time
DisjunctiveNet.jl | 2671   2671  1m16.0s
Testing DisjunctiveNet tests passed
```

Only after these two passes was `scripts/verify_sage_gradient.jl` rerun. Thus
the current Phase 4 output has the required evidence ordering.

