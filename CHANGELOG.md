# Changelog

## 0.2.0-alpha.1 - 2026-11-14

- Added exact geometry-aware smoothing based on optimizer switching normals.
- Added ordinary importance weighting that preserves the compact-`q` target.
- Added `SAGEConfig` and diagnostics for sampling, certificates, solver work and
  effective sample size.
- Added certificate ablations and explicit accounting of geometry branch solves.
- Added distribution, equation-21, estimator-parity, feasibility and validation
  tests.
- Added the pre-registered SAGE alpha novelty benchmark and paper-facing log.
- Added Kaggle `alpha` reproduction mode.

This is a research alpha. Its pass token supports the controlled mechanism
claim described in `docs/SAGE_ALPHA.md`; it is not a production-stability or
broad-superiority claim.
