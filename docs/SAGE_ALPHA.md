# SAGE Core research alpha

Version: `0.2.0-alpha.1`

SAGE Core is a proposal-faithful research implementation for differentiating
through an exact projection onto a finite union of convex branches. This alpha
packages the first complete controlled mechanism test; it is not a production
API or a claim of universal empirical superiority.

## What is specific to SAGE

The contribution tested by this release is the coupling of four components:

1. exact feasible branch-wise forward projection;
2. a fixed-reference local derivative plus a switching residual (proposal
   equations 17-18);
3. safe ball and per-sample certificates that remove provably irrelevant
   global searches (equations 11-12);
4. an adaptive proposal constructed from optimizer switching normals
   `y_j^* - y_i^*` (equations 19-21), with exact `q/p_z` weighting.

Perturbation, score estimators, importance sampling and variance reduction are
not claimed as individually new. The research claim is narrower: solver
certificates and solver-derived switching geometry are used together to decide
where exact computation is needed and where smoothing samples are valuable,
without changing the compact-`q` smoothed objective.

## Minimal API

```julia
using DisjunctiveNet

loss(y) = sum(abs2, y .- target)
loss_gradient(y) = 2 .* (y .- target)

config = SAGEConfig(
    radius = 0.5,
    n_samples = 64,
    beta = 2,
    sampling = :geometry,
    geometry_mixture = 0.5,
    geometry_focus = 0.9,
    use_ball_certificate = true,
    use_sample_certificate = true,
)

estimate = sage_gradient(model, z, loss, loss_gradient, config)
```

Key diagnostics are stored in `estimate`: sampling mode and geometry status,
switching samples, global searches, reference and competitor branch solves,
geometry construction solves, effective sample size and exact-feasibility
records.

For ablations, use the same samples and change only the certificate flags:

```julia
base = SAGEConfig(radius=0.5, n_samples=64,
    use_ball_certificate=false, use_sample_certificate=false)
cert = SAGEConfig(radius=0.5, n_samples=64)
geometry = SAGEConfig(radius=0.5, n_samples=64, sampling=:geometry)
```

## Reproduction

```bash
julia --project=. scripts/benchmark_sage_novelty_alpha.jl
julia --project=. test/runtests.jl
```

The novelty gate is pre-registered in
`docs/research_log/2026-11-14_sage_alpha_novelty.md`. Its recorded result is
`SAGE_ALPHA_NOVELTY=PASS`: geometry retained 44.96% of independent-sampling
variance and 62.75% of antithetic-sampling variance; certificates plus geometry
reduced complete branch solves by 30% versus the uncertified base. All backend
outputs remained logically feasible and parity errors were zero.

The Kaggle notebook supports `RUN_MODE = 'alpha'` for this gate and
`RUN_MODE = 'full'` for all accepted gates plus the full test suite.

## Alpha limitations

- The controlled novelty benchmark is one-dimensional and analytical. It
  isolates the mechanism but is not a broad task distribution.
- The Iris experiment is a five-seed pilot whose paired confidence intervals
  include zero; it is not statistically decisive.
- Geometry construction can add work. The estimator reports it explicitly; in
  the recorded ablation geometry used more global searches than certificate-only
  SAGE but fewer than the uncertified base.
- `sage_gradient` is the research API. The existing Flux projection layer does
  not automatically replace its backward pass with loss-aware SAGE.
- A defensible publication claim still requires a systematic literature review,
  native-disjunction datasets, stronger baselines and independent reruns.

The architecture and equation numbering follow the proposal by Bui Trong Duc;
the repository implementation, tests and benchmark logs are the auditable
record of what this alpha currently supports.
