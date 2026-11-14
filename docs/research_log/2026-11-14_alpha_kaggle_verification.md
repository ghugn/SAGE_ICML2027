# SAGE Core alpha Kaggle verification

Date: 2026-11-14

Status: **PASS**

## Reproduction identity

```text
run mode:       alpha
git commit:     32432546939609b415e29aa828702eb2b7655b91
release tag:    v0.2.0-alpha.1
Julia:          1.13.3
platform:       Linux x86_64, four CPUs
alpha elapsed:  28.399 seconds
```

The notebook cloned and tested the exact commit referenced by the alpha tag.
Foundation, Phase 6-8 and full-test stages were intentionally skipped because
`RUN_MODE = 'alpha'` isolates the pre-registered novelty gate. Those earlier
gates were separately reproduced and recorded in their own logs.

## Proposal-density preservation

```text
importance-weighted mass error:          0.00034145130320484274
importance-weighted mean error:          0.0010920642449677158
maximum second-moment error:              0.00037023589311055427
density gate:                             PASS
```

This verifies that geometry-aware proposal sampling plus exact `q/p_z` weights
continues to estimate the original compact-`q` target rather than silently
changing the smoothed objective.

## Matched-budget fidelity and variance

At matched `N=64`, 100 repetitions and all three analytical centers, the
geometry estimator passed the relative-bias gate. Aggregate results were:

```text
geometry / independent-q variance:       0.44963382794630163
geometry / antithetic-q variance:         0.6274598220542835
geometry / independent-q RMSE:            0.6717461855255034
geometry / antithetic-q RMSE:              0.7943894886212125
fidelity gate:                             PASS
variance gate:                             PASS
```

The antithetic comparator is the important control: geometry retains a
substantial variance advantage even after comparing against paired `u,-u`
sampling rather than only against independent Monte Carlo.

## Certified backend ablation

```text
Base / Cert / Geometry global searches:       320 / 90 / 123
Geometry global-search reduction vs Base:          61.5625%
Base / Cert / Geometry complete branch solves: 640 / 410 / 448
Geometry complete-solve reduction vs Base:          30.0000%
mean geometry effective sample size:              58.5988 / 64
maximum Base/Cert gradient parity error:          0.0
maximum geometry/direct backend parity error:     0.0
unsafe or logically infeasible outputs:           0
compute gate:                                     PASS
```

Geometry deliberately spends more global searches than certificate-only SAGE
(`123` versus `90`) to sample informative switching regions. Once geometry
construction is included, it still performs 30% fewer complete branch solves
than uncertified SAGE-Base and substantially reduces gradient variance.

## Conclusion and claim boundary

The exact alpha tag independently reproduces the controlled mechanism claim:
optimizer-derived switching geometry improves matched-budget gradient
efficiency, exact importance correction preserves the target, and certified
adaptive computation retains exact logical safety while reducing complete
solver work.

This run does not prove first-ever literature novelty or broad real-data
superiority. Those remain paper-level questions requiring a systematic
literature comparison and native multi-dataset experiments.
