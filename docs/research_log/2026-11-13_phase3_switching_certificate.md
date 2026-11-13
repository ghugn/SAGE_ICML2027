# Phase 3 Research Log — Switching Certificates

Date: 2026-11-13  
Scope: branch stability inside an Euclidean perturbation ball  
Priority: zero false stable claims first; coverage and speed second

## Entry 0 — Hypothesis and safety contract

Let `k0` be the globally certified branch winner at center `yhat`. For any
competitor group `G`, the distance function to a closed feasible set is
1-Lipschitz. Therefore a sufficient condition for `k0` to remain strictly
closer throughout `norm(delta) <= rho` is

```text
lower_distance(G, yhat) - upper_distance(k0, yhat)
    > 2*rho + certificate_tol.
```

The implementation uses the analytic geometric lower bound from Phase 2 and a
constructive accepted center-branch point as the winner upper bound. It never
uses a continuous-QP primal objective as a certified competitor lower bound.

Safety semantics:

- `:stable` means every competitor group passed the strict sufficient test;
- `:not_certified` is inconclusive and does not assert that a switch exists;
- `:center_uncertified` refuses to reason from a non-global center result;
- `:center_infeasible` reports that no center winner exists.

This is deliberately conservative. The research target is a useful balance
between exactness and speed, not a target pruning percentage.

## Entry 1 — Complete competitor partition

Added `src/certificates/competitor_set.jl`. All complete branches except the
winner are partitioned into disjoint prefix subtrees: at depth `d`, a group
matches the winner on earlier rules, differs at rule `d`, and leaves later
rules unassigned. For binary rules the group sizes are
`2^(m-1), 2^(m-2), ..., 1`, summing to `2^m - 1`.

The implementation checks this accounting invariant at runtime. The public
`competitor_set(certificate)` returns only groups whose stability condition was
inconclusive. This is the focused input intended for Phase 4 switching-aware
gradient work.

## Entry 2 — Certificate implementation and safeguards

Added `src/certificates/branch_margin.jl` with:

- `SwitchingCertificate` diagnostics;
- `switching_certificate`;
- `certifies_stability` and `is_certified`;
- branch counts, group margins, center time, and certificate-only time.

Safeguards include:

1. the center search must be globally `:optimal`;
2. the returned point must pass original-logic and selected-branch checks;
3. a reused center result must have an objective consistent with the supplied
   `yhat`, preventing common stale-result reuse;
4. ties use a strict inequality and are never stable-certified;
5. overlapping zero-distance branches remain competitors;
6. negative/non-finite radii are rejected.

The certificate establishes branch identity, not invariance of the active
linear constraints inside that branch. Phase 4 must still handle within-branch
active-set changes when defining differentiability.

## Entry 3 — Test matrix

Added `test/test_switching_certificate.jl`. Controlled checks cover:

- analytic 1D stable, wide-radius, and exact-tie cases;
- complete/disjoint competitor accounting under multiple rule orders;
- an exhaustive 2D perturbation grid;
- 40 seeded random near-boundary centers/radii with exhaustive 1D grids;
- overlapping disjuncts;
- ordinary convex models with no competing branch;
- infeasible and node-limit center results;
- stale center-result rejection.

The dedicated suite first passed 1,974 assertions, then gained the stale-result
assertion and passed as 1,975 Phase 3 assertions inside the complete package
regression below.

## Entry 4 — Verification benchmark

Command:

```powershell
julia --project=. scripts\verify_switching_certificate.jl
```

After compilation warm-up, the script reported:

```text
Benchmark dimension/branches: 8/256
Center search time: 0.1508006 seconds
Certificate-only median time (21 runs): 3.2e-6 seconds
Certificate internal time: 2.6e-6 seconds
Stable-pruned competitor branches: 255
Competitive branches retained: 0
Minimum distance margin: 0.6999997454415838
Required margin: 0.040000001
2D exhaustive grid points: 197
Switches after stable claim: 0
Near-tie status: not_certified
Near-tie competitive branches: 3
SWITCHING_CERTIFICATE_VERIFICATION=PASS
```

Interpretation: on a clearly separated center the certificate overhead is tiny
relative to the already-computed center search. On the symmetric near-tie it
correctly retains all three competitor branches instead of forcing a pruning
decision. The 255/255 stable removal is an illustrative separated case, not a
coverage objective or distribution-free claim.

### Corrected script issue

The first verification-script run failed because Julia global-loop soft scope
made integer counters ambiguous. Counters were changed to `Ref` values. This
was a script-only failure; no certificate result was accepted from that run.

## Entry 5 — Limitations and next decision

- The current theorem and API cover Euclidean perturbation balls only.
- The geometric group bound is safe but may be loose for intersections of
  several constraints, reducing certificate coverage near complicated faces.
- Solver outputs and residual checks are numerical certificates under the
  configured tolerances, not symbolic exact arithmetic.
- Empirical grids are regression audits, not the proof; the Lipschitz distance
  inequality is the proof mechanism.
- Phase 3 does not yet implement a gradient estimator.

### Complete package regression

The first `Pkg.test()` attempt could not create its sandbox under the default
Windows `%TEMP%` because that path was outside the writable workspace. A second
attempt at a parent `tmp` directory was also denied. Pointing `TEMP` and `TMP`
to a temporary directory inside the package allowed the normal isolated package
test environment to run. Pkg emitted a non-fatal registry/SSH permission
warning and used the already installed manifest dependencies.

```text
Test Summary:     | Pass  Total     Time
DisjunctiveNet.jl | 2534   2534  1m24.6s
Testing DisjunctiveNet tests passed
```

Phase 3 exits with the complete regression passing. Phase 4 should use
the certificate as a fast path: apply the fixed-branch implicit gradient only
when stable; otherwise evaluate the retained competitor set or use a clearly
defined switching correction. It must never silently treat `:not_certified` as
stable.

