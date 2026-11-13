# Phase 8 Iris data

Source: UCI Machine Learning Repository, Iris dataset, DOI
[`10.24432/C56C76`](https://doi.org/10.24432/C56C76).

- Creator: R. A. Fisher.
- Repository record: https://archive.ics.uci.edu/dataset/53/iris
- Retrieved: 2026-11-13.
- Upstream file: `bezdekIris.data` from the official `iris.zip` archive.
- Upstream SHA-256: `0FED2A99DB77EC533A62DC66894D3EC6DF3B58B6A8F3CF4A6B47E4086B7F97DC`.
- Committed LF-normalized SHA-256:
  `36F668D1CBC29A8C2C1128C5D2F0D400FA04ED4DC62D12246F44CE9360360CC0`.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The file contains 150 plants, four continuous measurements in centimetres and
one of three species labels. Phase 8 uses sepal length/width as network inputs
and petal length/width as regression outputs. Species labels are used only to
fit the three output-space disjunct boxes from each training split and to report
branch-identification accuracy. Validation and test observations never set box
bounds or normalization statistics.
