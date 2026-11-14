# Phase 11 research log — certified multiscale SAGE

Date: 2026-11-14  
Status at registration: hypothesis and mathematical contract frozen before
implementation results

## Motivation retained from Phase 10

The fixed-radius 3-by-3 job-shop pilot produced 40% branch switching but no
global-search reduction: all 120 SAGE perturbations required a global search.
Small-radius diagnostics did not cross a branch boundary, while `rho=0.50`
crossed boundaries but defeated the whole-ball certificate. Phase 10 remains
unchanged as the negative baseline.

Phase 11 tests whether the same target smoothing objective can allocate samples
between a certified local region and optimizer-derived boundary directions
without introducing a biased two-objective gradient.

## Mathematical contract

The target remains the compact proposal distribution `q(u)` on the unit ball
and the physical target radius remains

```text
rho_boundary = 0.50.
```

Equation (18) is still evaluated at

```text
w = z + rho_boundary * u.
```

Phase 11 does **not** add gradients computed for different smoothing
objectives. Instead it changes only the importance proposal. Let `s` be the
certified local scale and define

```text
p_local(u) = s^(-d) q(u/s),  ||u|| < s,
p_focus,j(u) / q(u) = (1 + kappa * dot(n_j,u))^2 / Z.
```

The combined proposal is

```text
p_z = w_base*q + w_local*p_local + w_geometry*mean_j p_focus,j,
w_base + w_local + w_geometry = 1.
```

Every complete Eq. (18) sample is multiplied by the ordinary importance
weight `q(u)/p_z(u)` and the estimator divides by the declared sample count,
not by the random weight sum. The base component has positive mass, so support
is preserved. No gradient is propagated through the certificate, scale,
normals or proposal parameters.

The certified physical local radius is determined only from the Phase 3 strict
margin:

```text
rho_local = min(0.45 * max(minimum_margin, 0), 0.95 * rho_boundary),
s = rho_local / rho_boundary.
```

The factor `0.45 < 0.5` leaves a 10% safety buffer relative to the sufficient
condition `minimum_margin > 2*rho_local + certificate_tol`. If no positive
margin exists, the local mixture weight is reassigned to the base component;
the estimator never invents a certified local ball.

Frozen nominal mixture weights are:

```text
w_base = 0.20,
w_local = 0.40,
w_geometry = 0.40,
beta = 2,
kappa = 0.9.
```

These weights are fixed before observing method quality. A margin-only audit
may report local-scale coverage but may not select weights using train/test
loss or method rankings.

## Proposal alignment

1. Exact feasible branch-wise forwards retain equations (7)-(10).
2. The safe distance margin and competitor set retain equations (11)-(12).
3. The full local plus switching residual integrand retains equations
   (17)-(18) at one fixed target radius per estimator call.
4. Optimizer-derived switching normals retain equations (19)-(21).
5. The new scaled-local component is a variance/computation proposal with an
   exact density ratio; it does not change the target objective.

## Pre-registered ablation

The controlled Job-Shop data, network, oracle, initialization, five seeds,
train/IID/OOD splits, six fine-tuning updates, target radius, `N=4`, learning
rate and method ordering remain identical to Phase 10.

Primary comparison:

- frozen Phase 10 SAGE-Geometry (`fixed_geometry`);
- Phase 11 certified multiscale geometry (`multiscale_geometry`).

Black-box, hard-local and STE Phase 10 results remain contextual controls. The
implementation must also expose component counts, certified local radii,
importance ESS, global searches, complete branch solves and exact feasibility.

## Gates to freeze after implementation smoke

Before the five-seed pilot is observed, the smoke must be used only to validate
density, exact feasibility, nonzero local-component coverage, backend parity
and a feasible runtime. Final numerical thresholds for safety, fidelity, ESS,
learning and quality-or-computation will then be appended and frozen. Failed
smokes and pilot failures remain in this log.

No Phase 11 advantage exists at registration time. It must not be described as
a repair or improvement until the frozen ablation is run.

## 2026-11-14 - retained margin-only audit failure

After the exact multiscale density implementation passed its targeted tests, a
margin-only audit inspected the 30 Phase 10 training centers selected by the
five seeds and six-update schedule. It did not inspect task loss or method
rankings.

```text
positive certified local radii: 0/30
minimum-margin min/median/mean/max:
  -0.5393 / -0.2424 / -0.2532 / -0.0533
certified local-radius min/median/mean/max:
  0 / 0 / 0 / 0
```

Therefore the registered sampler would reassign its local mass to the base
component at every audited center and collapse to a fixed-radius geometry
proposal. It is invalid to report a certified-local Phase 11 experiment with
these margins.

The cause is structural: the existing analytic lower bound takes axis-box
distance plus the maximum single non-axis constraint violation. Job-shop
precedence and no-overlap branches are coupled difference constraints, so the
single-row bound is too weak for their intersections. The next implementation
step is restricted to strengthening this **certified lower bound**, not tuning
the sampler against quality. A nonnegative analytic dual combination of all
selected linear inequalities is permitted: every dual-feasible vector gives a
mathematically safe lower bound on projection distance. Solver primal
objectives remain forbidden as pruning certificates. All existing exhaustive
pruning audits must continue to report zero unsafe prunes.

## 2026-11-14 - certified dual tightening and second margin audit

Implemented a solver-independent dual lower bound for every partial linear
branch. Cyclic coordinate ascent constructs nonnegative inequality
multipliers, but convergence is not assumed: weak duality makes every iterate
a safe projection-objective lower bound. A scale-aware floating-point guard is
subtracted before pruning. The previous row-wise/axis bound remains and the
maximum of the two safe bounds is used.

Targeted lazy-search and switching-certificate tests passed, including seeded
exhaustive oracle audits with zero unsafe prunes. On a coupled two-row example,
the dual combination raises the certified objective from the single-row value
`0.25` to the exact value `0.50`.

Repeating the same margin-only audit after this tightening gave:

```text
positive certified local radii: 22/30
minimum-margin min/median/mean/max:
  -0.1240 / 0.01108 / 0.00771 / 0.11704
certified local-radius min/median/mean/max:
  0 / 0.00498 / 0.00913 / 0.05267
local-scale median/mean:
  0.00997 / 0.01826
```

The local component is now valid at most centers but occupies a very small
fraction of the target ball. This predicts a real computation/ESS trade-off
and is retained rather than hidden.

## 2026-11-14 - implementation smoke and frozen pilot gates

The one-seed smoke used the frozen Job-Shop family, three fine-tuning updates,
`N=4`, and target `rho=0.50`:

| method | OOD loss | global searches | branch solves | ESS |
|---|---:|---:|---:|---:|
| black-box | 0.08852 | 12 | 28 | n/a |
| strengthened fixed geometry | 0.07317 | 8 | 57 | 3.949 |
| certified multiscale geometry | 0.07278 | 5 | 54 | 1.994 |

Multiscale reduced global searches by `58.3%` versus black-box and `37.5%`
versus strengthened fixed geometry. Two of three estimator calls had positive
local radii; the proposal drew `3/6/3` base/local/focused samples. Maximum
logical violation was `1.39e-16`. This is runtime/mechanism validation only.

The five-seed pilot gates are frozen before observation:

1. every trained IID/OOD output has CSAT `1.0`, maximum violation at most
   `1e-6`, and zero unresolved/logical failures;
2. strengthened fixed geometry reduces global searches by at least `10%`
   versus matched black-box;
3. multiscale reduces global searches by at least `20%` versus strengthened
   fixed geometry;
4. multiscale complete branch solves are at most `1.05x` strengthened fixed
   geometry branch solves;
5. at least `50%` of multiscale estimator calls have a positive certified
   local radius and at least one local sample is drawn;
6. multiscale mean ESS is at least `0.40N`;
7. multiscale improves the exact pretrained OOD loss and stays within `1.10x`
   both black-box and strengthened fixed-geometry OOD loss;
8. exact compact-target density tests, importance identity and exhaustive
   pruning audits remain passing.

Paired OOD-loss intervals and wall time are reported but are not superiority
gates. No threshold above will be changed after the five-seed run.

## 2026-11-14 - frozen five-seed pilot result

The pilot completed all 15 matched runs and passed every pre-registered gate:

| method | OOD loss | seconds | global searches | branch solves | ESS |
|---|---:|---:|---:|---:|---:|
| black-box | 0.111687 | 2.226 | 120 | 196 | n/a |
| strengthened fixed geometry | 0.107959 | 2.023 | 69 | 488 | 3.964 |
| certified multiscale geometry | 0.107084 | 2.020 | 51 | 460 | 2.822 |

Mechanism results:

- strengthened fixed geometry reduced global searches by `42.5%` versus
  black-box, showing that the safe dual lower bound repairs the certificate
  coverage failure observed in Phase 10;
- multiscale reduced global searches by `57.5%` versus black-box and by
  `26.09%` versus strengthened fixed geometry;
- multiscale used `460/488 = 0.943x` the complete branch solves of fixed
  geometry;
- `21/30 = 70%` estimator calls had a positive certified local radius;
- the proposal drew `37/37/46` base/local/focused samples;
- mean certified local radius was `0.009114` at target radius `0.50`;
- mean ESS was `2.822/4`, above the frozen `0.40N` gate;
- every trained IID/OOD output had CSAT `1.0`, with maximum logical violation
  `2.22e-16` and zero unresolved/logical failures.

Quality interpretation:

- multiscale improved black-box mean OOD loss by `0.004603` and strengthened
  fixed geometry by `0.000876`;
- paired multiscale-minus-fixed 95% CI:
  `[-0.002979, 0.001228]`;
- paired multiscale-minus-black-box 95% CI:
  `[-0.012754, 0.003548]`.

Both intervals cross zero. Phase 11 therefore supports a computation and
certificate-coverage claim, not an accuracy-superiority claim. Although local
wall time was slightly lower than black-box, multiscale still used more
complete branch solves (`460` versus `196`) because reference and geometry
projections are counted. The supported result is specifically fewer expensive
global searches and modestly fewer complete solves than fixed geometry.

Final token:

```text
PHASE11_CERTIFIED_MULTISCALE=PASS
```

## 2026-11-14 - packaging and regression verification

- Added deterministic unit tests for the scaled-ball density ratio, complete
  multiscale mixture, importance identity and target mass/moments.
- Added a coupled-constraint dual-bound test and retained all exhaustive
  zero-unsafe-pruning audits. A further `242/242` random coupled partial-branch
  audit confirmed every dual bound stayed below its exact best completion.
- Added an end-to-end Phase 11 Job-Shop test covering certified local radius,
  proposal component accounting, ESS, zero unresolved samples and exact IID/OOD
  feasibility.
- Added `phase11` Kaggle mode, made it the default, routed CSV/trajectory/JSON
  artifacts into the reproduction bundle and included the accepted gate in
  notebook `full` mode.
- Updated the README without converting the non-significant accuracy result
  into a superiority claim.
- Targeted Phase 11 tests passed `12/12`.
- Full package regression passed `3481/3481` in `2m22.6s` on Julia 1.13.3.

Generated local smoke and pilot folders remain ignored. The committed runner
is ready for independent Kaggle reproduction.
