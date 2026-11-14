# Phase 10 Kaggle verification

Date: 2026-11-14  
Status: negative boundary independently reproduced  
Commit: `6b4f3fa1a5270795f771a685100d7753d9f8d053`

## Environment and execution contract

The public Kaggle notebook ran in `phase10` mode on Julia 1.13.3, four CPU
cores and no GPU. Dependency instantiation completed in `3.083s`; the Phase 10
stage completed in `1835.492s` (about 30 minutes 35 seconds).

The notebook required the exact recorded token
`PHASE10_JOBSHOP_DIAGNOSTIC=FAIL`. The benchmark emitted that token, exited
with code zero under the explicit diagnostic-reproduction mode, and the final
notebook summary reported:

```text
phase10_jobshop_diagnostic = NEGATIVE_BOUNDARY_REPRODUCED
```

Earlier phases and the full test suite were `SKIPPED` because this was the
isolated Phase 10 mode. Their skipped status is intentional and is not a test
failure.

## Reproduced configuration

- jobs/machines/operations: `3/3/9`;
- native machine-order disjunctions/complete branches: `9/512`;
- seeds/train/IID/OOD: `5/16/12/12`;
- fine-tuning updates/samples/radius: `6/4/0.5`;
- mean/min serial-dispatch makespan gap: `0.931685/0.612903`;
- maximum oracle violation: `2.423e-8`;
- black-box switching rate: `0.4`.

## Reproduced quality result

| method | OOD task loss | OOD makespan gap | OOD CSAT |
|---|---:|---:|---:|
| pretrained exact | 0.117850 | 0.593818 | 1.0 |
| hard-local | 0.105711 | 0.559440 | 1.0 |
| STE | 0.104546 | 0.554819 | 1.0 |
| black-box | 0.111687 | 0.577379 | 1.0 |
| SAGE-Cert | 0.106903 | 0.565260 | 1.0 |
| SAGE-Geometry | 0.108973 | 0.574251 | 1.0 |

The paired SAGE-Geometry minus black-box OOD task-loss difference was
`-0.002714`, with 95% CI `[-0.009422, 0.003994]`. The differences versus STE
and hard-local were positive and their intervals also crossed zero. Kaggle
therefore reproduces the local conclusion: SAGE learns and is numerically
better than black-box on this slice, but this experiment does not establish a
superiority claim.

## Reproduced computation result

| method | mean training seconds | global searches | complete branch solves |
|---|---:|---:|---:|
| hard-local | 22.143 | 0 | 2,874 |
| STE | 21.976 | 0 | 2,830 |
| black-box | 36.564 | 120 | 13,993 |
| SAGE-Cert | 37.093 | 120 | 25,352 |
| SAGE-Geometry | 36.398 | 120 | 25,384 |

SAGE-Geometry ESS was `3.959/4`. Safety, adaptation, switching, learning,
matched-quality and ESS gates passed. The compute and frozen
quality-or-compute gates failed because SAGE performed all `120` global
searches and improved black-box OOD loss by only `2.43%`, below the
pre-registered `10%` threshold.

Wall time was slower than the local Windows run, while task metrics and solver
work counts reproduced to numerical precision. This separates platform speed
from the algorithmic conclusion.

## Conclusion

This is a successful reproduction of a negative experimental boundary, not a
failed notebook run and not an accepted SAGE application gate. It supports the
recorded Phase 10 diagnosis: branch count and discontinuity alone do not make
fixed-radius certificate/geometry SAGE computationally advantageous. The
fixed result remains the baseline for a future pre-registered adaptive-radius
ablation.

Kaggle artifacts were written beneath:

```text
/kaggle/working/sage_reproduction_logs/20261114_073137_phase10/
  phase10_jobshop_results/phase10_runs.csv
  phase10_jobshop_results/phase10_trajectories.csv
  phase10_jobshop_results/phase10_summary.json
```
