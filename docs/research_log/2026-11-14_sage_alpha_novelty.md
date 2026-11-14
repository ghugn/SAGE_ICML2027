# SAGE research-alpha novelty gate

Date: 2026-11-14

Status at registration: implementation and acceptance criteria frozen before
the final benchmark run. Results must be appended without deleting failures or
relaxing thresholds.

## Research question

Does proposal Component 4 add a demonstrably structure-aware contribution
beyond generic stochastic smoothing: can optimizer-derived switching normals
reduce estimator variance while exact importance weighting preserves the same
compact-`q` objective, and can the existing certificates still reduce complete
solver work after all geometry overhead is counted?

This gate does not claim that perturbation, score gradients, control variates,
branch-and-bound, or importance sampling are individually new. The candidate
novelty is their optimizer-specific coupling:

1. exact feasible projection over a finite union of convex branches;
2. fixed-reference local differentiation plus switching residual, equations
   (17)-(18);
3. certified zero-residual and competitor removal, equations (11)-(12);
4. proposal directions computed from `y_j^* - y_i^*`, equations (19)-(21);
5. exact `q/p_z` correction, so geometry changes sampling efficiency but not the
   smoothed objective.

## Frozen geometry proposal

For unit switching normals `n_j`, the focused component is

```text
p_j(u) / q(u) = (1 + kappa * dot(n_j, u))^2 / Z,
Z = 1 + kappa^2 / (d + 2*beta + 2).
```

The normalization is analytic because the compact polynomial-ball density is
isotropic and

```text
E_q[(dot(n_j,u))^2] = 1 / (d + 2*beta + 2).
```

The final proposal is

```text
p_z = (1 - alpha) q + alpha * mean_j p_j,
alpha = 0.5, kappa = 0.9, beta = 2.
```

Since `alpha < 1`, `p_z` covers the complete support of `q`. Rejection sampling
draws exact focused samples and the full equation-(18) integrand is multiplied
by the ordinary importance weight `q/p_z`. The estimator divides by `N`, not by
the random sum of weights. No gradient is propagated through the adaptive
proposal.

Normals are computed at the center from accepted competitor branch projections.
Degenerate or unavailable geometry falls back to `q`. Geometry branch solves
are reported separately and included in complete branch-solve comparisons.

## Frozen benchmark

Command:

```text
julia --project=. scripts/benchmark_sage_novelty_alpha.jl
```

Default statistical design:

```text
controlled centers:        -0.3, -0.2, -0.1
radius:                     0.5
target:                     1.5
fidelity repetitions:      100
samples per estimate:      64
backend repetitions:       5
backend samples:            64
moment samples:             50,000
```

The one-dimensional union `[-2,-1] union [1,2]` has an analytical compact-`q`
gradient on these slices:

```text
dJ/dz = -11.25 * (1 - (-z/rho)^2)^2.
```

Three matched-`N` estimators are compared: independent isotropic `q`,
antithetic isotropic `q`, and geometry-aware importance sampling. Antithetic
sampling is retained as a strong existing control variate.

The backend ablation compares:

- `SAGE-Base`: equations (17)-(18), no ball/sample certificate;
- `SAGE-Cert`: identical `q` samples plus both certificates;
- `SAGE-Geometry`: certificates plus equations (19)-(21) proposal sampling.

## Acceptance criteria fixed before the final run

All gates must pass:

1. Importance-weighted mass error at most `0.02`, mean norm at most `0.02`,
   and maximum coordinate second-moment error at most `0.015`.
2. Maximum geometry relative bias at most `0.10` across all analytical slices.
3. Aggregate geometry/isotropic variance ratio at most `0.75`.
4. Aggregate geometry/antithetic variance ratio at most `0.85`.
5. Per-slice geometry/isotropic variance ratio at most `0.90`.
6. Base/certificate gradient parity and geometry/backend direct parity below
   `1e-8` for identical realized samples.
7. Zero unsafe or logically infeasible backend outputs.
8. At least `25%` fewer perturbed global searches and `15%` fewer complete
   branch solves than `SAGE-Base`, including geometry branch solves.

Passing this controlled gate supports a research-alpha mechanism claim. It is
not sufficient for an ICML novelty or broad empirical-superiority claim; the
same ablation must later be tested on multi-branch and native-disjunction tasks.

## Files introduced or changed

- `src/gradients/geometry_sampling.jl`: exact proposal density, rejection
  sampler, switching-normal construction and diagnostics;
- `src/gradients/sage_estimator.jl`: importance-weighted sampling, certificate
  ablations, geometry counters and `SAGEConfig`;
- `test/test_geometry_sampling.jl`: distribution, geometry, exact-feasibility
  and validation tests;
- `scripts/benchmark_sage_novelty_alpha.jl`: frozen novelty gate.

## Final run (criteria unchanged)

Status: **PASS** on Julia 1.13.3. No acceptance threshold above was changed
after seeing these results.

Importance-target validation with 50,000 proposal draws:

```text
weighted mass error:                    0.0003414513
weighted mean-norm error:               0.0010920642
maximum second-moment error:            0.0003702359
```

All three controlled slices passed the analytical-gradient bias gate. The
largest geometry relative bias was `0.02552`. Aggregated across slices:

```text
geometry / independent-q variance:      0.4496338279
geometry / antithetic-q variance:        0.6274598221
geometry / independent-q RMSE:           0.6717461855
geometry / antithetic-q RMSE:            0.7943894886
```

The antithetic comparison is important: the gain is not merely the familiar
variance reduction obtained by pairing `u` and `-u`. It remains after comparing
against that stronger control variate under matched sample count.

Complete backend ablation, including proposal construction:

```text
maximum Base/Cert same-sample parity error:       0.0
maximum geometry/backend direct parity error:     0.0
unsafe or infeasible outputs:                      0
Base / Cert / Geometry global searches:       320 / 90 / 123
Base / Cert / Geometry complete branch solves: 640 / 410 / 448
Geometry global-search reduction vs Base:       61.5625%
Geometry branch-solve reduction vs Base:         30.0000%
mean geometry effective sample size:          58.5988 / 64
```

Geometry uses more global searches than certificate-only SAGE (`123` versus
`90`) because it deliberately samples closer to switching boundaries. That is
the expected computation/variance tradeoff, not a hidden win: it still uses far
less complete work than uncertified SAGE-Base while substantially lowering
variance.

## Interpretation and claim boundary

This run validates a concrete SAGE research-alpha contribution: exact
branch-feasible forward projection, certified adaptive computation and
optimizer-derived switching geometry can be coupled into one unbiased
compact-`q` gradient estimator. The density, estimator parity, logical safety,
variance and complete-work claims were all tested separately.

It does **not** establish that every ingredient is new, that SAGE is the first
possible geometry-aware estimator, or that it wins on arbitrary real tasks.
Those claims require a complete literature review and multi-dataset native
disjunction experiments. The alpha therefore exposes diagnostics and ablations
instead of presenting the controlled pass as a finished paper result.
