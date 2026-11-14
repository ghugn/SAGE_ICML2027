# DisjunctiveNet / SAGE Development and Verification Roadmap

Updated: 2026-11-14

> **Numbering note.** Sections 3-9 below preserve the original implementation
> roadmap. The executed research phases later became more granular: Phase 5 is
> the discontinuity-learning gate, Phase 6 the matched-budget multi-branch
> gate, Phase 7 controlled scaling, Phase 8 the Iris pipeline pilot, and the
> research alpha adds proposal Component 4. The dated research logs are the
> canonical audit trail; the current Phase 9 plan is recorded at the end of
> this document.

## 1. Reproducible local environment

The workspace contains two verified official Julia runtimes:

- Primary: `../.tools/julia-1.13.3/bin/julia.exe`
- Compatibility runtime: `../.tools/julia-1.10.13/bin/julia.exe`
- Project-local depot: `../.julia_depot`

PowerShell setup from the package directory:

```powershell
$env:JULIA_DEPOT_PATH = (Resolve-Path '..\.julia_depot').Path
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. -e 'using Pkg; Pkg.instantiate()'
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. -e 'using Pkg; Pkg.test()'
```

Initial baseline on Julia 1.13.3 with registered DiffOpt 0.6.2:

- 171 tests pass.
- 5 tests error during reverse differentiation.
- The common failure is a singular KKT factorization in DiffOpt.
- Forward modeling and projection tests pass, but passing them does not prove
  satisfaction of the original nonconvex logical rules.

Stabilized Phase 0 baseline:

- 214/214 tests pass with both direct execution and `Pkg.test()`.
- Singular reverse KKT systems retry with a project-owned LSQR fallback.
- `check_feasibility` verifies outputs against the original Boolean rules.
- Examples report logical feasibility and maximum original-rule violation.
- Detailed evidence is recorded in
  `docs/research_log/2026-11-13_phase0_baseline.md`.

The DiffOpt fork currently referenced by the README contains a regularized KKT
solve, but its current `master` is not source-compatible with Julia 1.13.3. On
Julia 1.10.13 it loads, but the current Flux/KernelAbstractions dependency stack
does not precompile reliably. Dependency compatibility must therefore be fixed
and pinned before treating gradient results as reproducible.

## 2. Definition of correct output

An output is accepted only when all of the following hold:

1. Solver status is acceptable (`OPTIMAL`, or a documented tolerance-bounded
   status).
2. Box and global constraint residuals are at most `feasibility_tol`.
3. Every active original disjunctive rule is satisfied by at least one of its
   disjuncts. Membership in a convex hull is not sufficient.
4. For exact branch projection, the reported incumbent is accompanied by a
   valid lower bound and `upper_bound - lower_bound <= optimality_tol`.
5. The result contains no `NaN` or `Inf`, and tie-breaking is deterministic.

For the SAGE gradient path, correctness additionally requires:

1. Zero unsafe prunes against an exhaustive oracle on all controlled tests.
2. The stability certificate is evaluated on distances, not directly on
   squared-distance objectives.
3. The estimator is compared against the gradient of the same smoothed
   objective, using the same distribution and radius.

## 3. Phase 0 - Stabilize the current baseline

Status: complete on 2026-11-13. This stabilizes the current convexified
baseline; it does not turn the relaxation into an exact logical projector.

### Work

- Pin a tested Julia and package matrix in `Manifest.toml` or CI.
- Make the regularized DiffOpt backend compatible with the selected Julia
  version, or implement the KKT regularization in a maintained local backend.
- Fix the layer pullback so it forwards `num_dnf_rules` and `rule_ordering`.
- Make `mode` and `gradient` select real implementations rather than only store
  configuration labels.
- Add `is_logically_feasible(model, y; atol)` and use it in every projection
  test and example.
- Add an explicit test demonstrating that convex-hull membership can differ
  from original logical feasibility.

### Exit gate

- All 176 original tests plus the new baseline-verification tests pass on the
  primary runtime.
- Every example reports solver status, original-rule feasibility, primal
  residual, and objective value.
- Forward and backward use exactly the same formulation configuration.

## 4. Phase 1 - Exact branch oracle

Status: implementation complete on 2026-11-13; analytic and seeded randomized
validation pass. Cross-solver MIQP comparison remains optional pending access to
a compatible MIQP solver, and is recorded as an external-validation item rather
than silently assumed.

Current regression result: 375/375 tests pass in a fresh `Pkg.test()`
environment.

Implement an exhaustive reference solver for small problems before adding any
pruning.

Suggested modules:

```text
src/exact/branch_types.jl
src/exact/branch_projection.jl
src/exact/exhaustive_oracle.jl
src/validation/feasibility.jl
test/test_exact_oracle.jl
```

For each leaf branch, solve

```text
minimize 0.5 * ||y - z||^2
subject to y in the selected original branch.
```

Record branch ID, feasibility, projected point, distance, solver status and
residuals. Use deterministic tie-breaking.

### Exit gate

- Analytic interval and box cases match closed-form projections.
- Random 2D/3D cases match a MILP/global reference within tolerance.
- Returned outputs satisfy the original logic in 100% of feasible cases.
- Empty and contradictory rule sets return an explicit status, never identity
  fallback disguised as success.

## 5. Phase 2 - Lazy certified branch search

Status: implementation and controlled validation complete on 2026-11-13.
Lazy search matches the exhaustive oracle with zero audited unsafe prunes on
the current deterministic/random suite. A separated eight-rule benchmark
pruned 255/256 branches and was 6.86x faster after warm-up. This is a controlled
result, not a general 99% pruning guarantee.

Current regression result: 2671/2671 tests pass in a fresh `Pkg.test()`
environment after the initial Phase 4 implementation.

Do not create the full Cartesian product before pruning.

Each search node contains a partial disjunct assignment, remaining rules, a
valid lower bound and optional warm-start information. Use CNF/partial-DNF
relaxations as lower-bound problems and a feasible leaf as the incumbent upper
bound. Expand nodes best-first by lower bound.

Prune a node for center projection only when its certified lower bound cannot
beat the incumbent under the configured tolerance.

Suggested modules:

```text
src/search/branch_node.jl
src/search/node_relaxation.jl
src/search/lazy_branch_search.jl
src/search/search_cache.jl
test/test_lazy_search.jl
```

### Exit gate

- Lazy search and exhaustive oracle return the same winning objective and a
  logically feasible point on every controlled random instance.
- Unsafe prune count is exactly zero.
- Report explored nodes, pruned groups, leaf solves, relaxation solves, memory
  and wall time separately.
- Demonstrate a real wall-time improvement, not only a high branch-prune rate.

## 6. Phase 3 - Switching certificates

Status: implementation and controlled validation complete on 2026-11-13.
The implementation partitions all non-winning branches into disjoint prefix
groups, stable-prunes only groups satisfying the strict distance margin, and
retains every inconclusive group as an explicit competitor set. Correctness is
prioritized over coverage: `:not_certified` is a safe fallback, not a switch
claim.

For center winner `k0`, use distance bounds to construct the competitor set.
A group is stable-pruned only when

```text
lower_distance(group) - upper_distance(k0) > 2 * rho + certificate_tol.
```

If the solver exposes `0.5*d^2`, convert objective bounds to distance bounds
before applying the certificate.

Suggested modules:

```text
src/certificates/branch_margin.jl
src/certificates/competitor_set.jl
test/test_switching_certificate.jl
```

### Exit gate

- Exhaustive perturbation grids in 1D/2D never find a switch after a stable
  certificate.
- Random adversarial tests near ties and overlaps produce zero false
  certificates.
- Certificate coverage and certificate overhead are both reported.

## 7. Phase 4 - Switching-aware gradients

Status: initial proposal-faithful polyhedral implementation complete on
2026-11-13. It implements fixed-branch implicit differentiation, a C1
compact-support ball density, residual switching correction, stable-certificate
fast path, and exact search over the retained competitor set. Boundary-focused
importance sampling was subsequently completed and verified in research alpha
`v0.2.0-alpha.1`; see the dated alpha novelty and Kaggle logs.

Acceptance ordering: Phase 4 metrics are gated by
`scripts/verify_foundations.jl` and the complete package regression. Prototype
output produced before that combined gate is classified only as a smoke test.

Implement the method in two independently testable components:

1. Within-branch implicit gradient for a fixed branch.
2. Residual switching correction for a precisely defined smoothed objective.

Start with isotropic sampling and fixed `rho`. Add antithetic samples and common
random numbers before geometry-focused importance sampling. Do not average
projected outputs; only aggregate losses or gradient estimators.

Suggested modules:

```text
src/gradients/local_implicit.jl
src/gradients/smoothing_distribution.jl
src/gradients/switching_correction.jl
src/gradients/sage_estimator.jl
test/test_gradient_fidelity.jl
```

### Exit gate

- The 1D two-interval example matches its analytic smoothed gradient.
- Local gradients pass central finite differences away from active-set changes.
- Full estimator bias, variance, norm error and cosine similarity are measured
  against a high-accuracy reference.
- No silent zero-gradient or identity-gradient fallback is permitted.

## 8. Phase 5 - Flux training integration

Add a structured result type carrying the projected output, branch ID, bounds,
certificate status, residuals and diagnostics. Keep search diagnostics out of
the differentiable parameter set.

Add deterministic RNG handling, batching by active-rule signature, solver
model caching and warm starts across perturbations.

### Exit gate

- A complete forward/backward/update step passes with finite gradients.
- Training is reproducible from a recorded seed.
- Loss decreases on the interval benchmark from a deliberately wrong branch.
- Inference always uses the hard exact center projection.

## 9. Phase 6 - Scalability and research validation

Evaluate controlled slices over dimension, number of rules, disjunct count,
branch separation, overlap, near-boundary rate and smoothing radius.

Required baselines:

- Exhaustive branch projection.
- CNF, fixed-order partial-DNF and full DNF where possible.
- Hard local KKT gradient.
- Straight-through estimator.
- Black-box stochastic smoothing / perturbed optimizer.
- SAGE without certificates, without switching correction and without focused
  sampling.

Primary plots must use matched global-solver-call and wall-time budgets.

### Go/no-go criteria

Proceed to large applications only if:

- original-rule satisfaction is 100% on feasible inputs;
- unsafe prune count remains zero;
- exact-search optimality gaps meet tolerance;
- the estimator improves gradient fidelity or training under matched cost; and
- certificate/pruning overhead is smaller than the work it saves.

Do not claim a fixed pruning percentage such as 99% until it is demonstrated
across controlled regimes with confidence intervals and total wall-clock cost.

## 10. Executed status and Phase 9

Completed and independently recorded:

- exact branch-wise forward, equations (7)-(10);
- switching stability and competitor pruning, equations (11)-(12);
- fixed-reference local plus residual estimator, equations (13)-(18);
- optimizer-normal geometry proposal with exact importance correction,
  equations (19)-(21);
- controlled discontinuity learning, matched-budget multi-branch tests,
  dimension/branch scaling and a five-seed Iris end-to-end pilot;
- SAGE Core `v0.2.0-alpha.1`, independently reproduced on Kaggle.

Phase 9 now implements proposal Roadmap Stage 4 as a native-disjunction
cooling-control reconstruction. The public upstream repository does not contain
the numeric cooling experiment, so all unavailable constants are explicitly
declared rather than described as reproduced. The frozen five-seed pilot
compares hard-local, STE, black-box, certified SAGE and geometry-aware SAGE
under matched initialization and perturbation budgets. It passes its safety,
native-disjunction, learning, matched-quality, compute and ESS gates: all exact
forwards have 100% IID/OOD CSAT and geometry SAGE reduces perturbed global
searches by 65.04% while matching mean black-box OOD quality. The paired MSE
interval includes zero, so superiority is not claimed.

PBMC3k and robot/energy applications remain later tasks. Nonlinear convex or
intrinsically nonconvex extensions must not be silently claimed from the current
polyhedral backend.

See `docs/research_log/2026-11-14_pre_phase9_audit.md` for the evidence map and
entry conditions, and `docs/research_log/2026-11-14_phase9_cooling_control.md`
for the retained solver failure, repair, frozen protocol and result.

## 11. Continuous verification commands

Fast checks during development:

```powershell
$env:JULIA_DEPOT_PATH = (Resolve-Path '..\.julia_depot').Path
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. test\runtests.jl
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. examples\basic_example.jl
```

Release gate:

```powershell
$env:JULIA_DEPOT_PATH = (Resolve-Path '..\.julia_depot').Path
& '..\.tools\julia-1.13.3\bin\julia.exe' --project=. -e 'using Pkg; Pkg.test(; coverage=true)'
```

Archive the Julia version, Manifest hash, seed, tolerances, solver versions and
full diagnostic summary with every benchmark result.
