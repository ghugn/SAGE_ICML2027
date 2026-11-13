# Research Log Convention

Every development phase gets one dated Markdown file in this directory. This is
the paper-facing audit trail, not a substitute for source-control history.

Each material step must record:

1. hypothesis or failure being tested;
2. files and semantics changed;
3. exact reproducible command;
4. quantitative result and acceptance criterion;
5. limitations, negative results, and paper-relevant interpretation.

Logs:

- [`2026-11-13_foundation_gate.md`](2026-11-13_foundation_gate.md) — combined
  pre-SAGE gate for baseline, original-logic validation, and exact oracle.
- [`2026-11-13_phase0_baseline.md`](2026-11-13_phase0_baseline.md) — baseline
  stabilization and original-logic output verification.
- [`2026-11-13_phase1_exact_oracle.md`](2026-11-13_phase1_exact_oracle.md) —
  exhaustive exact branch oracle used as pruning ground truth.
- [`2026-11-13_phase2_lazy_search.md`](2026-11-13_phase2_lazy_search.md) —
  best-first lazy search, certified geometric bounds, and exhaustive prune audit.
- [`2026-11-13_phase3_switching_certificate.md`](2026-11-13_phase3_switching_certificate.md) —
  perturbation-ball branch stability, conservative competitor sets, and grid audits.
- [`2026-11-13_phase4_sage_estimator.md`](2026-11-13_phase4_sage_estimator.md) —
  proposal-faithful local/residual gradient decomposition and fidelity tests.
- [`2026-11-13_phase5_discontinuity_learning.md`](2026-11-13_phase5_discontinuity_learning.md) —
  controlled go/no-go test of whether SAGE escapes a dead fixed-branch gradient.
- [`2026-11-13_github_migration.md`](2026-11-13_github_migration.md) — clean
  GitHub migration, Kaggle runner, and pre-push verification record.
- [`2026-11-13_phase6_multibranch_matched_budget.md`](2026-11-13_phase6_multibranch_matched_budget.md) —
  controlled multi-branch gradient fidelity, certification, pruning, and
  matched-budget efficiency gate.
- [`2026-11-13_workspace_consolidation.md`](2026-11-13_workspace_consolidation.md) —
  records why the GitHub repository is the DisjunctiveNet project and restores
  a single canonical local working tree.
- [`2026-11-13_unified_full_kaggle_verification.md`](2026-11-13_unified_full_kaggle_verification.md) —
  records the single-revision Kaggle run in which every gate and all 2671
  regression tests passed before Phase 7.
- [`2026-11-13_phase7_scaling_smoke.md`](2026-11-13_phase7_scaling_smoke.md) —
  pre-registers and records the honest scaling failure: correctness and 50%
  global-search reduction persist, but fixed-budget fidelity and total backward
  time do not yet scale to dimensions 4 and 8.
- [`2026-11-13_phase7_completion.md`](2026-11-13_phase7_completion.md) —
  completes the controlled scaling gate with tighter safe geometry, exact lazy
  competitor search, independent reference checks, dimension-aware antithetic
  sampling, and a proposal-faithful variance-or-computation decision.
- [`2026-11-13_phase7_kaggle_verification.md`](2026-11-13_phase7_kaggle_verification.md) —
  independent four-CPU Kaggle reproduction of commit `96a3f74`, including all
  Phase 7 acceptance metrics and the timing-variability interpretation.
- [`2026-11-13_phase8_iris_pilot.md`](2026-11-13_phase8_iris_pilot.md) —
  pre-registers and records the first real-data pilot: train-only Iris species
  boxes, exact feasible forwards, matched training budgets and paired 5-seed
  comparisons of hard-local, STE, black-box smoothing and SAGE.

