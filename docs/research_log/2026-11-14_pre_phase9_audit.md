# Pre-Phase-9 proposal and research-log audit

Date: 2026-11-14

Status: **ready for a native-disjunction Phase 9**, subject to obtaining or
reconstructing the original cooling-control task definition and baseline.

## Purpose

This audit prevents Phase 9 from drifting away from the SAGE proposal or
mistaking controlled alpha evidence for a completed applications program. It
checks formula coverage, experimental coverage and note completeness before any
new benchmark thresholds are chosen.

## Proposal-to-implementation map

| Proposal item | Implementation evidence | Primary log | Status |
|---|---|---|---|
| Exact branch projections, Eq. (7)-(10) | `src/exact/*`, exhaustive oracle and deterministic ties | Phase 1 | PASS for finite polyhedral branches |
| Exact scalable winner search | `src/search/*`, lower bounds and exhaustive prune audit | Phase 2 | PASS on controlled instances |
| Stability and competitor set, Eq. (11)-(12) | `src/certificates/*`, strict `2rho` margin | Phase 3 | PASS; conservative by design |
| Compact smoothed objective, Eq. (13)-(16) | `smoothing_distribution.jl` | Phase 4 | PASS for fixed `rho`, beta >= 2 |
| Local plus switching residual, Eq. (17)-(18) | `local_implicit.jl`, `switching_correction.jl`, `sage_estimator.jl` | Phases 4-6 | Analytical and Monte Carlo PASS |
| Switching geometry, Eq. (19)-(21) | `geometry_sampling.jl` | Alpha novelty gate | PASS on controlled analytical geometry |
| Exact `q/p_z` importance correction | geometry sampler plus weighted full Eq. (18) integrand | Alpha novelty gate | Density and parity PASS |
| End-to-end training procedure | controlled discontinuity and Iris scripts | Phases 5 and 8 | PASS as pipeline evidence |
| Exact feasible inference | lazy center projection and original-logic validator | Phases 1-8 | PASS on current supported models |

The implementation follows the proposal's finite-union core, but the current
backend is still polyhedral. Ellipsoids, SOCP branches, intrinsic nonconvexity
and trainable discontinuous rule activation are not implemented guarantees.

## Experimental-design coverage

| Proposal experiment | Evidence so far | Remaining gap |
|---|---|---|
| E1 controlled geometry | intervals and controlled polyhedral multi-branch grids | nonlinear convex geometry and wider native tasks |
| E2 gradient fidelity | analytical reference, high-budget reference, bias/variance/RMSE/cosine | application-level learning gradients |
| E3 certification/scalability | branch and dimension slices, complete solver accounting | larger native rule systems and memory curves |
| E4 reproduce DisjunctiveNet | stabilized package baseline only | cooling-control and PBMC3k task reproduction |
| E5 realistic hybrid applications | none | robot corridor or hybrid energy after E4 |
| E6 ablations/failures | local/STE/black-box/SAGE, certificate and sampling ablations; retained Phase 7 failure | fixed-radius sensitivity and native-task failure analysis |

Therefore the next proposal-aligned move is E4, not another claim that the Iris
species boxes constitute a sufficient application benchmark.

## Research-log completeness

Every scientific Phase 0-8 has a dated note containing its question or
hypothesis, code semantics, quantitative result, interpretation and limitation.
Phase 7 retains both the failed fixed-budget smoke test and the completed gate.
Phase 8 and the alpha have independent Kaggle records tied to exact commits.

The audit found one omission: the Phase 2 note reported the verification output
but did not preserve the exact command. The command has now been added. GitHub
migration and workspace-consolidation notes are administrative records and are
not treated as scientific phases.

## Phase 9 scope fixed by the proposal

Working title: **native-disjunction cooling-control reproduction and SAGE
comparison**.

Primary research question:

> On an inherited task whose constraints are natively disjunctive, can
> certified/geometry-aware SAGE match or improve learning quality relative to
> black-box smoothing and the original convexified path while preserving exact
> feasibility and reducing complete solver work under matched budgets?

Required sequence:

1. Recover and document the original task, split, preprocessing, rules, loss
   and published evaluation protocol. No SAGE result is accepted before the
   inherited baseline is reproduced or the discrepancy is documented.
2. Build one task adapter that creates the exact branch model and independently
   checks the original rules for every output.
3. Run a deterministic smoke test to measure scale and choose feasible budgets.
   Freeze seeds, radii, sample counts and acceptance criteria before the final
   comparison; do not tune thresholds on final results.
4. Compare at minimum hard-local, STE, matched black-box smoothing,
   certificate-only SAGE and geometry-aware SAGE. Include the original
   DisjunctiveNet convexified baseline when its objective/output semantics are
   compatible, and label incompatible surrogate objectives explicitly.
5. Use at least five paired seeds and report confidence intervals, per-seed
   trajectories and failure cases.

Required metrics:

- task loss and the inherited task metric;
- original-rule satisfaction and maximum violation;
- unresolved/infeasible input rate and branch/mode decisions;
- training progress versus total global solver calls and wall time;
- reference, competitor, geometry and relaxation/leaf solve counts;
- certificate/pruning rates, importance effective sample size and proposal
  overhead;
- paired quality differences and uncertainty across seeds.

Hard safety gates that can be fixed now:

- zero knowingly unsafe prunes;
- 100% original-rule satisfaction on resolved feasible outputs;
- finite gradients and losses with deterministic seeds;
- same-sample estimator/backend parity within the established numerical
  tolerance;
- all certificate and proposal overhead included in cost comparisons.

Quality and speed superiority thresholds must be frozen only after the inherited
baseline reproduction and smoke-budget measurement, because the task/data are
not yet present in this repository. Phase 9 must retain a negative result if
SAGE only matches quality, loses to STE, or geometry overhead exceeds its gain.

## Entry decision

The foundations required by proposal Roadmap Stages 1-3 are sufficiently
implemented and recorded to begin Phase 9. The missing input is application
definition/data, not another estimator component. Cooling control is the first
choice because the proposal explicitly prioritizes reproducing and extending
DisjunctiveNet before moving to robot or energy applications.
