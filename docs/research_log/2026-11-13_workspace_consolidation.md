# Workspace consolidation: one DisjunctiveNet working tree

Date: 2026-11-13

## Context

The research project has always been `DisjunctiveNet.jl-main`. The name
`SAGE_ICML2027` identifies the GitHub repository used to publish that project;
it does not identify a second implementation or an independent codebase.

A separate local `SAGE_ICML2027/` checkout was temporarily created during the
GitHub migration. This preserved the original files, but it also left two
physical working trees and made the canonical development location ambiguous.

## Consolidation decision

- Use `DisjunctiveNet.jl-main/` as the only canonical local development path.
- Preserve the complete Git history and the remote
  `https://github.com/ghugn/SAGE_ICML2027.git` in that working tree.
- Preserve the pre-consolidation directory as a recoverable backup under
  `DisNet/tmp/`; do not delete it during consolidation.
- Make all subsequent implementation, verification, notebook, and research-log
  changes only in `DisjunctiveNet.jl-main/`.

## Content audit before consolidation

The two working trees shared 72 relative files. All shared source and test files
were byte-identical. Differences were limited to the latest Phase 6 benchmark,
its research log, and updated notebook/README/index files in the Git-backed
working tree. The older tree additionally contained generated `.cov` coverage
files. Therefore, the Git-backed tree is the complete, current project state to
retain as the canonical `DisjunctiveNet.jl-main/` directory.

## Scientific impact

This is a workspace-layout correction only. It does not change the mathematical
formulation, solver behavior, SAGE estimator, benchmark results, or acceptance
criteria.
