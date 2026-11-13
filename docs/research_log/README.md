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

