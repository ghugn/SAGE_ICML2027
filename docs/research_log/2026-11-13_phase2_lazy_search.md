# Phase 2 Research Log — Lazy Certified Branch Search

Date: 2026-11-13  
Runtime: Julia 1.13.3  
Ground truth: `2026-11-13_phase1_exact_oracle.md`

## Entry 0 — Search objective and certification rule

### Objective

Implement a best-first search over partial disjunct assignments without
materializing the full Cartesian product. The result may be called exact only
when every unvisited branch belongs to a group removed by a valid bound or a
solver infeasibility certificate.

### Safe geometric lower bound

For a partial assignment, every completion lies inside:

- the model's box;
- every global constraint;
- every currently selected disjunct constraint.

Therefore the distance to any one of those supersets is a lower bound on the
distance to every completion. The implementation uses the maximum of:

1. exact Euclidean distance to the box;
2. distance to each selected/global halfspace or hyperplane.

The pruning objective bound is `0.5 * lower_distance^2`. This bound is weaker
than the full partial-assignment QP but is analytic and does not depend on an
unverified solver optimality gap.

### Important negative result

A direct probe of HiGHS through JuMP gave:

```text
QP optimum ~= 0
objective_bound(model) = 9
```

for a one-variable convex QP whose minimizer is interior. Consequently,
`objective_bound` is not used for certified QP pruning. The partial QP primal
solution is used only to prioritize nodes, detect solver-reported infeasibility,
and construct an early feasible incumbent. This distinction prevents a solver
API value with unclear semantics from becoming a false pruning certificate.

### Hard acceptance gates

1. Search result and exhaustive oracle have the same certified status and
   objective on controlled instances.
2. Returned output passes original-logic verification.
3. Every prune record is audited against stored exhaustive branch results.
4. Unsafe prune count is exactly zero.
5. A complete certified run accounts for every original branch as either a
   solved leaf or a member of a pruned group.

## Entry 1 — Implementation

### Search structure

- `src/search/branch_node.jl` defines nodes, a deterministic binary min-heap,
  search diagnostics, prune records, and audit results.
- `src/search/node_relaxation.jl` implements the analytic bound and the partial
  QP used only for priority/infeasibility/greedy seeding.
- `src/search/lazy_branch_search.jl` implements best-first expansion, strict
  bound pruning, deterministic tie handling, node limits, and full branch
  accounting.
- `src/search/pruning_audit.jl` checks each prune record against exhaustive
  branch results.

The search starts with a greedy feasible branch derived from the root relaxed
point. Its objective is an incumbent upper bound. A subtree is bound-pruned only
when its analytic lower bound is strictly greater than the incumbent objective
plus `prune_tol`; equality is explored so lexicographic ties remain correct.

### Smoke tests and corrected edge case

The Phase 0 counterexample produced the exact branch `[2, 1]` while solving two
of four leaves and pruning two. Exhaustive audit reported zero unsafe prunes.

An initial smoke test of a model with zero disjunctions exposed one extra zero
argument in the `LazySearchResult` constructor. The constructor was corrected,
and the ordinary convex case now returns one solved leaf and a certified exact
projection. This failure is retained as part of the implementation audit.

## Entry 2 — Controlled test matrix

Tests cover analytic lower bounds, the Phase 0 counterexample, symmetric ties,
seeded random 2D/3D comparisons, named rule ordering, global infeasibility,
zero-disjunction convex projection, node-limit uncertified termination, and a
controlled eight-rule separated instance designed to measure pruning coverage.

## Entry 3 — Verification results

### Reproduction commands

From the project root with the pinned Julia environment:

```powershell
julia --project=. scripts\verify_lazy_search.jl
julia --project=. test\runtests.jl
```

The first command emits `LAZY_SEARCH_VERIFICATION=PASS`; the second guards the
complete package against search regressions. The quantitative output below is
from the warmed verification run.

### Phase 2 assertions

The dedicated lazy-search suite passed 184 assertions covering:

- analytic lower-bound values;
- exact objective/status agreement with the exhaustive oracle;
- 20 seeded random 2D/3D instances;
- deterministic ties and alternative rule ordering;
- infeasible roots, ordinary convex models, and node-limit termination;
- complete prune-record coverage and zero unsafe prunes.

### Controlled pruning benchmark

After warming Julia, JuMP, and HiGHS, `scripts/verify_lazy_search.jl` reported:

```text
Total branches: 256
Leaf solves: 1
Relaxation solves: 8
Pruned branches: 255
Prune fraction: 0.99609375
Explored nodes: 8
Unsafe prunes: 0
Unsafe pruned branches: 0
All prunes recorded: true
Lazy time after warmup: 0.1229838 seconds
Exhaustive time after warmup: 0.8441028 seconds
Observed speed ratio exhaustive/lazy: 6.8635
LAZY_SEARCH_VERIFICATION=PASS
```

This benchmark deliberately places `yhat` well inside one of 256 separated
branches. It demonstrates that more than 99% pruning is achievable in a
favorable regime. It does not establish a distribution-free or worst-case 99%
guarantee. Near ties or overlapping branches can require many more leaf solves.

### Complete package regression

```text
Test Summary:     | Pass  Total     Time
DisjunctiveNet.jl |  559    559  1m13.5s
Testing DisjunctiveNet tests passed
```

## Entry 4 — Exit decision

Phase 2 is accepted for controlled research use. The center projection now has
a certified lazy exact path with explicit diagnostics and an exhaustive audit
mechanism. The next phase must not reuse the center-pruning condition directly
for perturbations: switching stability requires a distance-margin certificate
that accounts for the perturbation radius.

