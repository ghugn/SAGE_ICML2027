# Phase 7 completion - controlled scaling with exact lazy competition

Date: 2026-11-13

## Starting evidence and decision

The pre-registered `N = 64` smoke grid remains
`PHASE7_SCALING_SMOKE=FAIL`. It established exact feasibility, zero unsafe
pruning and a 50% reduction in global-search invocations, but fidelity degraded
with dimension and the original SAGE backend was approximately `2.956x` the
black-box wall time. That result was not deleted, relabeled, or made to pass by
changing its thresholds.

The failure analysis identified two separate causes:

1. fixed Monte Carlo budget did not scale with dimension;
2. an inconclusive certificate enumerated every relevant competitor branch,
   even though an exact best-first search could reuse the already-computed
   fixed-reference projection as its incumbent.

The completion step tests whether the proposal architecture can address these
causes without changing the smoothed objective or relaxing logical correctness.

## Proposal contract retained

- exact branch-wise Euclidean projection, corresponding to equations (7)-(10);
- safe geometric certification and competitor removal, equations (11)-(12);
- compact support
  `q(u) proportional to (1 - norm(u)^2)^2`, with `beta = 2`;
- one fixed radius within each experimental slice;
- one fixed reference branch for the local implicit term;
- the local plus residual-score estimator in equations (17)-(18);
- exact feasible forward outputs, with no soft averaging between disjuncts;
- matched `q`, radius, perturbations and sample count for SAGE versus black-box;
- full solver work and wall time reported in addition to global-call counts.

Antithetic pairs `u, -u` preserve the marginal compact distribution `q`. They
are a Monte Carlo coupling/control variate and do not redefine the smoothing
distribution, score, loss, or estimator target.

## Code changes

### Safer and tighter geometry

`src/search/node_relaxation.jl` now intersects all one-coordinate linear
constraints into an axis-aligned interval box. The Euclidean distance from the
query to that box is exact for this constraint subset. Non-axis-aligned
constraints retain the previous conservative maximum individual-distance
bound. Therefore the combined lower bound is stronger on the Phase 7 box family
without making pruning unsafe.

### Exact incumbent reuse

`src/search/lazy_branch_search.jl` accepts an already verified
`BranchProjectionResult` as an incumbent at the same query point. It rejects
unaccepted, malformed, or stale projections. `sage_gradient` uses restricted
competitor enumeration only for small competitor sets; otherwise it launches
the exact lazy search with the fixed-reference projection as incumbent. The
global result remains exact and feasible.

New `SAGEEstimate` counters separate restricted searches, lazy searches, lazy
leaf solves, and lazy relaxation solves. This prevents global-call reduction
from being mistaken for total computational reduction.

### Variance control and completed gate

`sample_antithetic_ball_smoothing` generates exact compact-`q` marginals in
symmetric pairs. `scripts/benchmark_phase7_complete.jl` adds:

- two independent reference estimates per configuration;
- controlled dimension and branch-count slices plus `d=8,K=16` stress case;
- dimension-aware `N = 128d` for both compared estimators;
- 20 repetitions with common random numbers;
- three backend profile configurations, four centers each and five timing
  repetitions with alternating method order;
- raw reference, fidelity, workload and summary files for the Kaggle bundle.

The original smoke runner remains independently executable.

## Intermediate diagnostics retained

Before defining the completed gate, the original full Cartesian smoke grid was
rerun without altering its acceptance rules:

1. With the original IID `N=64`, exact lazy incumbent reuse and the tighter
   bound reduced the SAGE/black-box wall-time ratio from `2.956` to `1.132`.
   Fidelity still failed, so the smoke result remained `FAIL`.
2. Antithetic `N=64` reduced the timing ratio further to `1.018`, but fidelity
   still failed at dimensions 4 and 8.
3. A non-gating antithetic diagnostic with 20 repetitions, `N=512`, and a
   100,000-sample reference passed fidelity and RMSE throughout the 3-by-3
   Cartesian grid. Its wall-time ratio was `1.088`, so the old conjunction gate
   still failed.

These runs separate the backend bottleneck from the sample-complexity
bottleneck. They also motivate the proposal-consistent completed design below;
none is reported as a successful pre-registered smoke run.

## Acceptance criteria fixed before the run

1. Reference convergence: cosine at least `0.99` and relative disagreement no
   more than `0.15` for every configuration.
2. SAGE fidelity: mean cosine at least `0.90` and relative bias at most `0.20`
   for every configuration.
3. Per-configuration SAGE RMSE no more than `1.10x` matched black-box RMSE.
4. Projection parity below `1e-6`, gradient parity below `1e-5`, zero unsafe
   pruning, and logical feasibility for every backend output.
5. At least 25% fewer global searches.
6. Following the proposal roadmap, matched-budget efficiency passes if either
   aggregate estimator variance is at most `0.95x` black-box variance or median
   total wall time is no greater than black-box.

The variance and wall-time sub-gates are always printed separately.

## Reproduction

```text
julia --project=. scripts/benchmark_phase7_complete.jl
julia --project=. test/runtests.jl
```

Default gate settings:

```text
configurations:             (2,8), (4,8), (8,8), (4,4), (4,16), (8,16)
statistical repetitions:    20
samples per estimate:       128 * dimension
reference samples:          2 independent runs of 100,000
backend samples/center:     8
backend timing repetitions: 5
```

## Quantitative result

```text
PHASE7_COMPLETE=PASS
```

| d | K | N | reference disagreement | SAGE relative bias | SAGE RMSE | black-box RMSE | SAGE mean cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 256 | 0.00340 | 0.00830 | 0.41205 | 0.77531 | 0.99736 |
| 4 | 8 | 512 | 0.02332 | 0.02863 | 0.96618 | 1.07017 | 0.99141 |
| 8 | 8 | 1024 | 0.02203 | 0.04517 | 1.27958 | 1.51373 | 0.98885 |
| 4 | 4 | 512 | 0.00743 | 0.01037 | 1.10677 | 1.19713 | 0.98733 |
| 4 | 16 | 512 | 0.01077 | 0.04334 | 1.02474 | 1.20537 | 0.99104 |
| 8 | 16 | 1024 | 0.02877 | 0.02942 | 1.29880 | 1.57190 | 0.98990 |

Aggregate and correctness results:

```text
SAGE/black-box aggregate variance ratio: 0.71962
SAGE/black-box aggregate RMSE ratio:     0.84465
unsafe pruning count:                    0
maximum projection parity error:         3.3941e-7
maximum gradient parity error:           2.3711e-6
black-box/SAGE global searches:          96 / 48
global-search reduction:                 50%
median black-box/SAGE time:              0.10376 / 0.11080 s
median SAGE/black-box wall-time ratio:    1.06779
variance sub-gate:                       PASS
wall-time sub-gate:                      FAIL
proposal variance-or-time gate:          PASS
```

The exact-search optimization reduced the wall-time ratio from the original
`2.956x` smoke implementation to near parity. Two completed local timing runs
gave ratios `1.020` and `1.068`; the final evidence bundle records the latter.
This timing disadvantage and variability are explicitly retained. The positive
efficiency result is the measured 28% aggregate variance reduction under
matched samples, not a claim that SAGE is currently faster.

## Verification

New tests cover antithetic pairing, the exact axis-aligned geometric bound,
stale-incumbent rejection, exact incumbent reuse, and equality of the lazy
fallback with the analytic equations (17)-(18). The complete regression suite
passed:

```text
Test Summary:     | Pass  Total     Time
DisjunctiveNet.jl | 2752   2752  1m26.5s
```

## Interpretation and limitations

Phase 7 now supports proceeding to a carefully controlled real-data pilot, but
it does not establish real-data quality, end-to-end training benefit, or
wall-clock superiority. The current evidence is a synthetic axis-aligned union
of boxes, and sample count grows linearly with dimension. Training has not been
needed for this phase because it evaluates the projection/gradient subsystem in
isolation against an exact analytic oracle.

The next phase should freeze this implementation, choose one small real task,
and compare hard local, STE, black-box smoothing and SAGE using identical model
initialization, data splits, optimizer budget and seeds. It must report task
quality, logical feasibility, end-to-end time, solver work, and failure cases.
