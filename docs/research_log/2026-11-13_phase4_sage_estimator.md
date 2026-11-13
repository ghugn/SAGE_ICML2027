# Phase 4 Research Log — SAGE Switching-Aware Gradient

Date: 2026-11-13  
Primary design source: Bui Trong Duc, *SAGE: Learning Across Discontinuities*,
proposal equations (7)-(18), especially (13)-(18).

## Entry 0 — Formula-to-code contract before implementation

This phase follows the proposal architecture rather than treating the existing
convexified DisjunctiveNet backward pass as the final method.

For center winner `k0`, define

```text
L(w)  = loss(P(w), target)
L0(w) = loss(project_branch(k0, w), target)
R0(w) = L(w) - L0(w)
Jrho(z) = E_u[L(z + rho*u)]
```

For a smooth density `q` supported in the unit ball and
`s_q(u) = -grad_u log q(u)`, implement proposal equation (17)/(18):

```text
grad Jrho(z) = E[grad L0(z + rho*u)]
             + (1/rho) E[R0(z + rho*u) * s_q(u)].
```

Non-negotiable semantic checks:

1. Every perturbed global output is an exact feasible branch projection.
2. Outputs are never averaged before applying the task loss.
3. The reference branch is the certified exact center winner and remains fixed
   throughout one estimator call.
4. A stable Phase 3 certificate sets `R0=0` without perturbed global solves.
5. An inconclusive certificate keeps the relevant competitor groups and solves
   them; it is never treated as stable.
6. `rho` is fixed within the estimator call. Adaptive schedules belong outside
   this objective unless their derivative terms are derived explicitly.
7. Uniform-ball samples must not be combined with the smooth-density score
   formula. The initial implementation will use
   `q(u) proportional to (1 - norm(u)^2)^beta`, `beta >= 2`, on the unit ball.

Planned modules:

```text
src/gradients/local_implicit.jl
src/gradients/smoothing_distribution.jl
src/gradients/switching_correction.jl
src/gradients/sage_estimator.jl
test/test_gradient_fidelity.jl
```

The first implementation targets the repository's current finite union of
polyhedra. Nonlinear convex branches and boundary-focused importance sampling
remain later extensions, matching the proposal hierarchy.

## Entry 1 — Implemented architecture

### Smooth compact-support perturbations

`src/gradients/smoothing_distribution.jl` implements

```text
q(u) proportional to (1 - norm(u)^2)^beta, beta >= 2,
s_q(u) = 2*beta*u / (1 - norm(u)^2).
```

The squared radius is sampled as `Beta(d/2, beta+1)` using two internal Gamma
samples, so sampling remains efficient without adding a distribution package.
The `beta >= 2` guard ensures the density extended by zero is C1 at the unit
ball boundary. The score is allowed to grow near the boundary; the code does
not make the invalid bounded-score assumption warned about in the proposal.

### Within-branch gradient

`src/gradients/local_implicit.jl` implements the polyhedral fixed-branch
projection Jacobian on an unchanged active face:

```text
J = I - A' * pinv(A*A') * A.
```

Here `A` contains active bound, global, and selected-branch constraint normals.
The task-loss gradient is multiplied by `J'`. This is an almost-everywhere
piecewise-linear derivative, not a claim of differentiability at active-set
changes.

### Exact relevant-competitor solve

`src/gradients/switching_correction.jl` solves the reference branch plus every
completion of the non-pruned competitor groups from Phase 3. The groups are
disjoint, tie-breaking remains lexicographic, and unresolved solver results
propagate as `:uncertified`. A branch-count guard prevents accidental unlimited
enumeration.

### Residual estimator

`src/gradients/sage_estimator.jl` implements proposal equation (18) per sample:

```text
sample_gradient = local_gradient + residual * score / rho.
```

If the Phase 3 certificate is stable, residual is exactly set to zero and no
perturbed global search occurs. Otherwise the global loss is evaluated on the
exact winner among the retained competitors. A supplied certificate is
recomputed and checked against the model/center/radius to reject stale or
mismatched certificates.

## Entry 2 — Implementation failures retained in the audit trail

1. The first compile failed because `Random` was only a test extra while the
   new runtime sampler imports it. `Random` was moved to package dependencies
   and `Pkg.resolve()` was run; no external package versions changed.
2. The first estimator parse failed because `local` is a Julia keyword. The
   internal variable was renamed `local_result` before any test result was
   accepted.
3. The extended verification script initially reused a mutable integer in a
   top-level loop, triggering Julia's soft-scope error. It was replaced by a
   `Ref` counter. This affected metrics collection only, not estimator code.

## Entry 3 — Correctness and fidelity tests

`test/test_gradient_fidelity.jl` covers:

- score values and the analytic radial second moment of the sampler;
- fixed-branch implicit gradients versus central finite differences;
- retained-competitor projection versus the exhaustive oracle at 21 points;
- stable-certificate fast path with zero perturbed global searches;
- exact residual, score, sign, and `1/rho` scaling for one switching sample;
- stale certificate rejection;
- deterministic quadrature for the proposal's two-interval example.

For `z=-0.4`, `rho=0.5`, target `1.5`, and
`q(u)=15/16*(1-u^2)^2`, switching occurs at `u>0.8`. Direct integration of
equation (17) gives `-1.458`. A 1,000-point midpoint-quantile evaluation gave
`-1.439202662827828`, absolute error `0.018797337172171913`, within the declared
quadrature tolerance `0.03`.

## Entry 4 — Reproducible verification and honest cost result

Command:

```powershell
julia --project=. scripts\verify_sage_gradient.jl
```

Result:

```text
Stable certificate status: stable
Stable perturbed global searches: 0
Stable reference branch solves: 32
Switching estimate: -1.439202662827828
Analytic smoothed gradient: -1.458
Absolute gradient error: 0.018797337172171913
Switching samples: 9/1000
Switching perturbed global searches: 1000
Switching competitor branch solves: 1000
MC repetitions/samples: 20/64
MC analytic reference: -10.368
MC mean estimate: -10.319649960744337
MC bias: 0.048350039255662836
MC variance: 11.36746695361668
MC RMSE: 3.2865531080802377
MC 1D cosine-sign match: 1.0
MC total perturbed global searches: 1280
SAGE_GRADIENT_VERIFICATION=PASS
```

Interpretation: the stable fast path realizes the proposal's solver-saving
claim in a certified regime. In the inconclusive regime, the current isotropic
baseline still performs one relevant global search per perturbation. The
non-trivial variance and RMSE are recorded rather than hidden; they motivate
the proposal's antithetic/control-variate and geometry-focused sampling work.
No general speedup or variance advantage is claimed at this stage.

### Complete package regression

`Pkg.test()` in the isolated package environment passed after implementation:

```text
Test Summary:     | Pass  Total   Time
DisjunctiveNet.jl | 2671   2671  59.8s
Testing DisjunctiveNet tests passed
```

The registry lookup emitted the already documented Windows SSH permission
warning and used the locked installed dependencies. `Pkg.resolve()` completed
without adding or removing packages.

## Entry 5 — Exit decision and limitations

The initial Phase 4 polyhedral core is accepted for controlled research use:
the local/residual decomposition, exact-feasible per-sample outputs, stable
fast path, and analytic gradient check are implemented. The following remain
open and must not be described as completed contributions:

- antithetic samples and common random numbers;
- geometry-aware boundary proposal and importance weights `q/p_z`;
- nonlinear convex branch implicit differentiation;
- cached/warm-started relevant subtree search;
- a Flux training adapter and end-to-end wrong-branch escape experiment;
- high-dimensional matched-budget bias/variance experiments.

The next implementation step should be proposal Component 4 or Phase 5 Flux
integration. Given the observed 1,000/1,000 global searches in an inconclusive
case, geometry-aware sampling plus matched oracle-budget evaluation is the more
direct research-risk test.

## Entry 6 — Post-foundation rerun

The user correctly required the baseline, logical validator, and exact oracle
to be verified together before interpreting Phase 4 output. Earlier Phase 4
results were reclassified as smoke tests. The following order was then
executed:

1. `scripts/verify_foundations.jl` — PASS;
2. isolated `Pkg.test()` — 2671/2671 PASS;
3. `scripts/verify_sage_gradient.jl` — PASS.

The post-gate Phase 4 numbers reproduced exactly, including analytic gradient
`-1.458`, estimate `-1.439202662827828`, absolute error
`0.018797337172171913`, stable-path zero perturbed global searches, and the
reported Monte Carlo bias/variance metrics. These are now the accepted
controlled Phase 4 outputs; the remaining limitations in Entry 5 still apply.

