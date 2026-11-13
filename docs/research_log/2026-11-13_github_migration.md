# GitHub Migration and Kaggle Reproduction Setup

Date: 2026-11-13

## Scope

The prior contents of `ghugn/SAGE_ICML2027` were replaced by the validated
Julia project at repository root. The previous Git state remains recoverable at
commit `06fa8c5e89fafa6bf3aad0e8213f21255bb2a783`.

Removed material included the legacy PyTorch implementation, generated plots,
old benchmark notebooks, and the duplicated nested Julia source tree. None of
those files are part of the new commit.

## Added reproducibility entry point

`SAGE_Kaggle_Reproducibility.ipynb` is a Python-kernel Kaggle runner that:

1. installs pinned Julia 1.13.3 from the official Julia distribution when
   Julia is absent;
2. clones the public repository and instantiates `Project.toml`;
3. requires pass tokens from the foundation gate, SAGE gradient verification,
   and Phase 5 discontinuity-learning test;
4. runs the complete Julia test suite;
5. saves the textual logs under Kaggle's writable working directory.

The notebook requests no GPU because the certified QP/search backend is a CPU
workflow. Internet access is required on the first run.

`Manifest.toml` is committed intentionally for exact Kaggle/paper
reproduction, even though Julia libraries often omit it for broader version
compatibility.

### Post-migration notebook audit

The runner was subsequently extended to record the platform, Python version,
CPU count and total memory; measure wall-clock time and exit/token status for
every Julia stage; refresh an existing Kaggle checkout from `origin/main`; and
produce `sage_reproduction_bundle.zip`. The archive contains the environment,
per-stage results, raw stdout logs and final acceptance summary needed for
remote diagnosis and paper bookkeeping.

## Local verification before commit

Commands were run against the migrated repository root with Julia 1.13.3:

```text
julia --project=. scripts/verify_foundations.jl
julia --project=. scripts/verify_sage_gradient.jl
julia --project=. scripts/benchmark_discontinuity_learning.jl
julia --project=. test/runtests.jl
```

Results:

- `FOUNDATION_VERIFICATION=PASS`;
- `SAGE_GRADIENT_VERIFICATION=PASS`;
- `DISCONTINUITY_LEARNING_GO_NO_GO=PASS`;
- full test suite: 2671 / 2671 passed;
- notebook JSON and all eight Python code cells parsed successfully.

`Pkg.test()` could not create a temporary environment under the local Codex
Windows sandbox (`Access is denied`), so the suite was run directly through
`test/runtests.jl`. The Kaggle notebook also uses the direct runner to avoid
this environment-specific temporary-directory failure.

## Interpretation

This migration removes unsupported legacy benchmark claims from the active
tree. The new repository status is explicit: foundations and the controlled
mechanism test pass, while a matched-budget nontrivial learning benchmark is
still future work.

