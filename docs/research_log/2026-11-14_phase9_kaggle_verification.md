# Phase 9 Kaggle verification

Date: 2026-11-14

Status: independently reproduced on the Kaggle Linux CPU environment.

## Purpose

Verify that the frozen Phase 9 cooling-control pilot runs from the public
repository rather than relying on local Windows state, stale artifacts or an
uncommitted environment.

## Exact revision and environment

- Git commit: `f5dba1e35174c515c188c5c6fbc81bdc3c5ec25e`
- Julia: `1.13.3`
- Platform: `Linux-6.18.55+-x86_64-with-glibc2.39`
- CPUs: `4`
- Memory: `32,869,296 KiB`
- Notebook mode: `phase9`
- Instantiate/precompile time: `689.679 s`
- Phase 9 stage time: `930.981 s`

Only the Phase 9 stage was requested. Foundation, Phase 4-8, alpha and full
tests therefore correctly reported `SKIPPED`; they were not failed or silently
omitted by the Phase 9 script.

## Frozen protocol

- five paired seeds;
- 250 training points, 500 IID and 500 OOD evaluation points per seed;
- five pretraining epochs and 250 exact-projection fine-tuning updates;
- compact proposal radius `rho=0.15` and `N=8` samples;
- the declared `CoolingConfig` printed and serialized by the runner;
- exact feasible forward projection for every constrained method.

## Kaggle result

| method | IID MSE | OOD MSE | IID/OOD CSAT | mean train seconds | perturbed global searches | complete branch solves |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.15077 | 0.23612 | 0.0504 / 0.0604 | 0 | 0 | 0 |
| pretrained exact | 0.05085 | 0.06426 | 1.0 / 1.0 | 0 | 0 | 0 |
| hard-local | 0.02879 | 0.03878 | 1.0 / 1.0 | 10.22 | 0 | 3,137 |
| STE | 0.02224 | 0.03004 | 1.0 / 1.0 | 9.55 | 0 | 1,870 |
| black-box | 0.02985 | 0.04385 | 1.0 / 1.0 | 22.31 | 10,000 | 17,102 |
| SAGE-Cert | 0.02831 | 0.04287 | 1.0 / 1.0 | 18.08 | 3,353 | 18,722 |
| SAGE-Geometry | 0.02791 | 0.04178 | 1.0 / 1.0 | 18.63 | 3,496 | 20,367 |

Additional output:

- maximum/mean OOD active rules: `5 / 2.216`;
- maximum oracle violation: `1.11e-16`;
- SAGE-Geometry minus black-box paired OOD-MSE difference:
  `-0.002068`, 95% CI `[-0.005719, 0.001583]`;
- SAGE-Cert/SAGE-Geometry global-search reductions:
  `66.47% / 65.04%`;
- geometry ESS: `7.67695 / 8`;
- safety, native-disjunction, learning, matched-quality, compute and ESS gates:
  all `PASS`;
- acceptance token: `PHASE9_COOLING_PILOT=PASS`.

## Cross-platform comparison

All task-quality metrics reproduce to floating-point precision. SAGE-Cert and
SAGE-Geometry global-search and complete-branch counts are identical to the
local run. Hard-local used one fewer complete branch solve (`3,137` versus
`3,138`) and black-box used two fewer (`17,102` versus `17,104`) on Kaggle.
This is a negligible solver-path difference across Windows and Linux, not a
task-output difference; the exact feasibility and MSE results remain stable.

Wall time is environment-dependent. Kaggle is slower than the recorded local
run, but SAGE-Geometry remains faster than matched black-box within the same
Kaggle environment (`18.63 s` versus `22.31 s` mean training time). It also
uses more complete local/geometry branch solves, so the supported efficiency
claim is reduction in expensive perturbed global searches and observed
end-to-end time, not reduction in every solver-work category.

## Interpretation

The independent run establishes reproducibility of the Phase 9 implementation,
its exact-feasibility contract and its frozen acceptance decision. It does not
establish statistically significant superiority over black-box smoothing: the
five-seed paired confidence interval includes zero. STE and hard-local remain
strong baselines and must stay in all paper-facing tables.

Phase 9 is still a paper-spec reconstruction because the upstream public
repository does not include the original cooling experiment's numeric
configuration. The evidence supports an application-ready research prototype,
not a claim that the original paper result has been exactly reproduced.

## Reproduction artifacts

Kaggle wrote:

```text
/kaggle/working/sage_reproduction_logs/20261114_025644_phase9/
```

including `phase9_runs.csv`, `phase9_trajectories.csv`,
`phase9_summary.json`, per-stage logs and the final notebook summary.
