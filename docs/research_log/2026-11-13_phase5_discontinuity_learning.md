# Phase 5 — Controlled Discontinuity-Learning Go/No-Go

Date: 2026-11-13

## Question

Before integrating SAGE into a larger Flux model, can the estimator actually
move a parameter through a projection discontinuity where the fixed-branch
implicit gradient is dead?

## Pre-registered experiment

- feasible output: `[-2,-1] union [1,2]`;
- target: `1.5`;
- initial parameter: `z=-0.4`, whose exact projection is the wrong left branch;
- compact smoothing density: `q(u) proportional to (1-u^2)^2` on `(-1,1)`;
- fixed radius `rho=0.5` (no adaptive objective changes);
- 30 gradient steps, learning rate `0.03`, 128 perturbations per stochastic
  step, 10 deterministic seeds;
- comparators: hard fixed-branch implicit gradient, STE, black-box score
  smoothing with a constant baseline, and proposal-faithful SAGE;
- report switch rate, first crossing step, final task loss and evaluation count.

The go criterion is deliberately narrow: SAGE must cross `z=0` in at least
80% of seeds, improve mean final loss over the dead hard-local baseline, and
the hard-local baseline must have zero crossings. Beating STE is not required;
STE is biased and is included as a practical heuristic comparator.

## Correctness guards

The fast repeated experiment uses the analytic projection for this one-
dimensional union. Before training it checks 61 points against certified
`lazy_branch_search`. It also checks the direct SAGE formula against
`sage_gradient` using the identical 256 perturbations.

## Reproduction

```text
julia --project=. scripts/benchmark_discontinuity_learning.jl
```

The script prints a machine-searchable final line:

```text
DISCONTINUITY_LEARNING_GO_NO_GO=PASS|FAIL
```

## Recorded local result

Environment: Julia 1.13.3, HiGHS backend, Windows x86-64.

| Method | Switch rate | Median crossing step | Mean final loss | Mean evaluation count |
|---|---:|---:|---:|---:|
| hard local | 0.0 | - | 6.25 | 30 |
| STE | 1.0 | 3.0 | 0.25 | 30 |
| black-box smoothing | 1.0 | 4.5 | 0.25 | 3840 |
| SAGE | 1.0 | 4.5 | 0.25 | 3840 |

Backend/direct SAGE parity error was `0.0`. All 10 SAGE seeds crossed the
discontinuity; the hard-local method crossed in none. The pre-registered result
was therefore `DISCONTINUITY_LEARNING_GO_NO_GO=PASS`.

This test does **not** show an advantage over STE or black-box smoothing: all
three crossed in this scalar setting. Its positive conclusion is only that the
SAGE correction repairs the specific dead-gradient failure it was designed to
repair. The next experiment must test variance, compute cost and learning
quality on nontrivial instances under matched budgets.

## Interpretation boundary

Passing this test only establishes a controlled scalar mechanism check. It
does not establish performance on learned representations, large disjunctions,
or real datasets. Those require the next staged benchmark with matched solver-
call and wall-clock budgets.

