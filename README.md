# DisjunctiveNet.jl

`DisjunctiveNet.jl` provides differentiable convexified projection layers for
neural networks with mixed logical-linear constraints.

> **Research status (2026-11-14):** the certified foundation, lazy search,
> switching certificate, proposal-faithful SAGE estimator, and controlled
> scalar, multi-branch, and controlled Phase 7 scaling gates are implemented.
> The original fixed-`N=64` Phase 7 smoke test remains an honest failure. The
> completed gate uses independent reference checks and dimension-aware
> antithetic sampling; it preserves exact feasibility, reduces global searches
> by 50%, and reduces aggregate estimator variance. Backend timing is currently
> near parity: local runs were slightly slower while the recorded Kaggle run
> gave a `0.996x` SAGE/black-box ratio. The first five-seed UCI Iris pilot now
> passes: SAGE has the lowest mean test MSE, 89% fewer perturbed global searches
> and a `0.667x` training-time ratio versus matched black-box smoothing. Its
> paired confidence intervals still include zero, so this is promising pilot
> evidence rather than broad real-data superiority. The research-alpha
> geometry gate now adds proposal Component 4: exact importance sampling along
> optimizer-derived switching normals. Against matched independent smoothing it
> retained 44.96% of the variance, and against antithetic smoothing 62.75%,
> while cutting complete branch solves by 30% versus the uncertified base. This
> establishes the proposed mechanism on controlled problems, not universal
> novelty or superiority. See [`docs/SAGE_ALPHA.md`](docs/SAGE_ALPHA.md) and
> [`docs/research_log`](docs/research_log) for the paper-facing audit trail.

## Kaggle reproduction

Upload and run [`SAGE_Kaggle_Reproducibility.ipynb`](SAGE_Kaggle_Reproducibility.ipynb)
with Internet enabled. The notebook installs a pinned Julia release, clones this
repository and instantiates dependencies. `RUN_MODE = "alpha"` runs the
pre-registered geometry-aware novelty gate; `RUN_MODE = "phase8"` runs the
five-seed UCI Iris pilot; `RUN_MODE = "phase7"` runs the controlled scaling
gate; `RUN_MODE = "full"` runs every accepted gate and the full regression suite. A GPU is not required
because the current certified search and QP backend are CPU workloads.

The package lets users define constraints of the form

```text
global linear constraints
+
(disjunct 1 OR disjunct 2 OR ...)
+
(disjunct 1 OR disjunct 2 OR ...)
+ ...
```

and automatically builds a differentiable projection layer that can be placed after a neural network, especially a Flux.jl model.

The intended workflow is:

```text
raw neural prediction
        ↓
differentiable disjunctive projection layer
        ↓
prediction satisfying the selected convexified formulation
```

The package is designed for research on differentiable constrained learning, neural-symbolic learning, and safe neural network prediction.

---

## Formulations

The layer supports three lifted convexified formulations.

| Formulation | Description |
|---|---|
| `:dnf` | Full DNF convex-hull formulation. Stronger but can grow combinatorially. |
| `:cnf` | One convex-hull block per disjunction. Scalable but generally weaker. |
| `:partial_dnf` | Expands a chosen subset of rules into DNF and keeps the rest as CNF. |

All three formulations are convexified projection layers. The projected output
is guaranteed to satisfy the selected convexified formulation. Any convex hull,
including full `:dnf`, can contain points outside the original nonconvex logical
set. Always use `check_feasibility(dm, y)` when original-rule feasibility is a
required output property.

### Exact small-instance oracle

For validation and pruning research, `exhaustive_project` projects onto every
complete original branch and returns the globally closest logically feasible
point when all branches are resolved:

```julia
result = exhaustive_project(dm, yhat)

result.status       # :optimal, :infeasible, or :uncertified
result.choices      # selected disjunct for each rule
result.y            # exact projected point
result.objective    # 0.5 * ||result.y - yhat||^2
result.report       # original-logic feasibility certificate
is_certified(result)
```

This is an exponential ground-truth oracle, not the scalable production path.
It generates branch assignments lazily and has a default `max_branches=100_000`
safety guard. The intended next layer is certified lazy branch search using the
convexified formulations as lower bounds.

### Lazy certified branch search

`lazy_branch_search` explores partial branch assignments best-first and removes
subtrees using analytic geometric lower bounds or solver-reported
infeasibility:

```julia
result = lazy_branch_search(dm, yhat)

result.status
result.y
result.choices
result.leaf_solves
result.relaxation_solves
result.pruned_branches
result.total_branches
```

For controlled small instances, every prune can be checked against the
exhaustive oracle:

```julia
oracle = exhaustive_project(dm, yhat)
audit = audit_pruning(result, oracle)

audit.unsafe_prunes
audit_passed(audit)
```

The analytic bound is intentionally conservative. A high pruning percentage is
instance-dependent and is not guaranteed in the worst case.

### Switching certificate over perturbation balls

`switching_certificate` tests whether the exact center winner remains the
winning branch for every input within an Euclidean ball of radius `rho`:

```julia
center = lazy_branch_search(dm, yhat)
certificate = switching_certificate(dm, yhat, rho; center_result = center)

certificate.status              # :stable or :not_certified
certificate.competitive_branches
competitor_set(certificate)      # groups that must still be considered
```

The certificate uses the strict distance condition
`lower_competitor - upper_winner > 2rho + certificate_tol`. A
`:not_certified` result is deliberately inconclusive: it never claims that a
switch exists. Reusing `center_result` avoids solving the center projection a
second time; it must have been computed for the same model and `yhat`.

Reproduce the Phase 3 correctness/overhead report with:

```bash
julia --project=. scripts/verify_switching_certificate.jl
```

### SAGE switching-aware gradient

The initial polyhedral SAGE estimator follows the residual decomposition in
the research proposal. The task loss is evaluated on each feasible output;
projected outputs are never averaged:

```julia
loss(y) = sum(abs2, y .- target)
loss_gradient(y) = 2 .* (y .- target)

estimate = sage_gradient(dm, z, rho, loss, loss_gradient; n_samples = 32)
estimate.gradient
estimate.certificate.status
estimate.perturbed_global_searches
estimate.switching_samples
```

The default smoothing density is proportional to
`(1 - norm(u)^2)^2` inside the unit ball. A stable certificate skips all
perturbed global searches but still averages the within-branch implicit
gradient at perturbed points, as required by the smoothed objective.

Reproduce the analytic and Monte Carlo fidelity report with:

```bash
julia --project=. scripts/verify_sage_gradient.jl
```

The research alpha exposes all proposal-aligned choices in a typed
configuration. Geometry mode uses exact ordinary importance weighting and
falls back to the original compact density when switching geometry is absent:

```julia
config = SAGEConfig(
    radius = 0.5,
    n_samples = 64,
    sampling = :geometry,
    geometry_mixture_weight = 0.5,
    geometry_focus_strength = 0.9,
)

estimate = sage_gradient(dm, z, loss, loss_gradient, config)
estimate.gradient
estimate.geometry_status
estimate.importance_effective_sample_size
```

Reproduce the pre-registered mechanism and ablation gate with:

```bash
julia --project=. scripts/benchmark_sage_novelty_alpha.jl
```

This compares independent compact-density smoothing, antithetic smoothing and
geometry-aware importance sampling, then separately ablates certificates and
counts geometry-construction work. The recorded gate reports
`SAGE_ALPHA_NOVELTY=PASS`. Its precise claim and limitations are documented in
[`docs/SAGE_ALPHA.md`](docs/SAGE_ALPHA.md).

Reproduce the Phase 6 controlled multi-branch fidelity and efficiency gate:

```bash
julia --project=. scripts/benchmark_multibranch_matched_budget.jl
```

This benchmark uses the same compact density, fixed radius and samples for
SAGE and black-box smoothing. It reports bias, variance, cosine similarity,
certificate/pruning rates, solver-call counts, wall time and original-logic
feasibility. Its result is a controlled gate, not a dataset-level claim.

Run the pre-registered Phase 7 dimension/branch-count scaling smoke grid:

```bash
julia --project=. scripts/benchmark_phase7_scaling.jl
```

The current recorded result is an honest `PHASE7_SCALING_SMOKE=FAIL`: exact
feasibility and safe pruning hold, but fixed-sample fidelity and complete SAGE
backward time do not yet scale to dimensions 4 and 8. See the Phase 7 research
log before changing its thresholds or benchmark geometry.

Run the completed proposal-aligned Phase 7 gate:

```bash
julia --project=. scripts/benchmark_phase7_complete.jl
```

This second gate does not replace the failed smoke test. It adds two independent
high-budget reference estimates, dimension-aware antithetic compact-`q`
sampling, controlled one-factor scaling slices, repeated timing, and the
proposal's variance-or-total-computation efficiency decision. It currently
reports `PHASE7_COMPLETE=PASS`; the separate wall-time sub-gate remains `FAIL`.

Run the Phase 8 real-data pilot:

```bash
julia --project=. scripts/benchmark_phase8_iris.jl
```

This uses the committed CC BY 4.0 UCI Iris data, train-only species-box
constraints, five stratified seeds and matched black-box/SAGE perturbations.
It saves per-seed runs, trajectories and solver-work diagnostics and currently
reports `PHASE8_IRIS_PILOT=PASS`. See the research log for the retained initial
failure, pointwise certificate proof, confidence intervals and limitations.

---

## Installation

The tested baseline uses registered DiffOpt 0.6.2. DisjunctiveNet keeps the
normal direct KKT solve and retries singular reverse systems with an LSQR
minimum-norm fallback owned by this package.

Install the package normally:

```julia
using Pkg

Pkg.add(url = "https://github.com/ghugn/SAGE_ICML2027.git", rev = "v0.2.0-alpha.1")
```

For local development:

```bash
git clone https://github.com/ghugn/SAGE_ICML2027.git
cd SAGE_ICML2027
julia --project=.
```

Then:

```julia
using Pkg
Pkg.instantiate()
Pkg.test()
```

---

## Basic idea

A neural network produces an unconstrained prediction:

```julia
yhat = backbone(x)
```

The projection layer maps it to a point satisfying the selected convexified rule set:

```julia
y = projection_layer(yhat)
```

The full constrained model is:

```julia
x -> backbone(x) -> disjunctive projection layer -> projected y
```

The projection is differentiable, so training can be performed using ordinary Flux/Zygote code.

---

## A first disjunctive model

Suppose the neural network predicts two outputs:

```text
y[1], y[2]
```

We want:

```text
0 <= y[1] <= 1
0 <= y[2] <= 1

global constraint:
y[1] + y[2] >= 0.8

rule 1:
y[1] <= 0.25 OR y[1] >= 0.75

rule 2:
y[2] <= 0.25 OR y[2] >= 0.75
```

This can be modeled as follows.

```julia
using DisjunctiveNet

# Create a disjunctive model with two output variables:
# y[1] and y[2].
dm = DisjunctiveModel(2)

# Set box bounds:
# 0 <= y[1] <= 1
# 0 <= y[2] <= 1
set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

# Add a global linear constraint.
# This constraint must hold in the projected output:
# y[1] + y[2] >= 0.8
add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

# Add rule 1:
# y[1] <= 0.25 OR y[1] >= 0.75
#
# The name :x_split is optional, but useful for partial-DNF ordering.
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)];
    name = :x_split,
)

# Add rule 2:
# y[2] <= 0.25 OR y[2] >= 0.75
#
# The name :y_split lets us refer to this rule later, for example in
# rule_ordering = [:x_split, :y_split].
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.25)],
    [LinearConstraint([0.0, 1.0], :>=, 0.75)];
    name = :y_split,
)
```

Inspect the user-facing model:

```julia
print_model(dm)
```

Inspect the lifted projection formulations:

```julia
print_projection_model(dm; formulation = :dnf)

print_projection_model(dm; formulation = :cnf)

print_projection_model(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

---

## Optional: JuMP-style linear constraints

For convenience, users can also write constraints using JuMP’s `@build_constraint` syntax. This avoids manually constructing `LinearConstraint` objects.

```julia
using JuMP
using DisjunctiveDifferentiableLayers

# Create a fresh model.
dm2 = DisjunctiveModel(2)

set_bounds!(
    dm2,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

# Get the JuMP variables associated with the model outputs.
y = output_variables(dm2)

# Add the global constraint:
# y[1] + y[2] >= 0.8
add_linear_constraint!(
    dm2,
    @build_constraint(y[1] + y[2] >= 0.8),
)

# Add rule 1:
# y[1] <= 0.25 OR y[1] >= 0.75
add_disjunction!(
    dm2,
    [@build_constraint(y[1] <= 0.25)],
    [@build_constraint(y[1] >= 0.75)];
    name = :x_split,
)

# Add rule 2:
# y[2] <= 0.25 OR y[2] >= 0.75
add_disjunction!(
    dm2,
    [@build_constraint(y[2] <= 0.25)],
    [@build_constraint(y[2] >= 0.75)];
    name = :y_split,
)
```

The explicit `LinearConstraint(...)` interface and the JuMP-style `@build_constraint(...)` interface create the same internal representation.

---

## Core API

### `DisjunctiveModel(n_outputs)`

Creates a model for a neural network output vector of length `n_outputs`.

```julia
dm = DisjunctiveModel(3)
```

This means the projection layer expects raw neural network predictions of length 3:

```julia
yhat = [yhat1, yhat2, yhat3]
```

and returns a projected vector:

```julia
y = [y1, y2, y3]
```

---

### `set_bounds!(model; lower, upper)`

Sets box bounds on the projected output variables.

```julia
set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)
```

This imposes:

```text
0 <= y[1] <= 1
0 <= y[2] <= 1
```

Finite bounds are strongly recommended because they are used by the convex-hull formulations.

---

### `LinearConstraint(a, sense, b)`

Represents a linear constraint of the form:

```text
a' y <= b
a' y >= b
a' y == b
```

depending on `sense`.

Examples:

```julia
LinearConstraint([1.0, 1.0], :>=, 0.8)
```

means:

```text
y[1] + y[2] >= 0.8
```

```julia
LinearConstraint([1.0, 0.0], :<=, 0.25)
```

means:

```text
y[1] <= 0.25
```

Supported senses are:

```julia
:<=
:>=
:(==)
```

---

### `add_linear_constraint!(model, ...)`

Adds a global linear constraint.

Global constraints are enforced across the full projection model. They are useful for constraints that should always hold, regardless of which disjunctive branch is active.

Coefficient-vector form:

```julia
add_linear_constraint!(
    dm,
    [1.0, 1.0],
    :>=,
    0.8,
)
```

Equivalent explicit object form:

```julia
add_linear_constraint!(
    dm,
    LinearConstraint([1.0, 1.0], :>=, 0.8),
)
```

Equivalent JuMP-style form:

```julia
using JuMP

y = output_variables(dm)

add_linear_constraint!(
    dm,
    @build_constraint(y[1] + y[2] >= 0.8),
)
```

---

### `add_disjunction!(model, disjuncts...; name = nothing)`

Adds a disjunctive rule.

Each argument is one disjunct, and each disjunct is a vector of linear constraints.

For example:

```julia
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)];
    name = :x_split,
)
```

means:

```text
y[1] <= 0.25 OR y[1] >= 0.75
```

A disjunct can contain multiple constraints:

```julia
add_disjunction!(
    dm,
    [
        LinearConstraint([1.0, 0.0], :>=, 0.4),
        LinearConstraint([1.0, 0.0], :<=, 0.6),
    ],
    [LinearConstraint([1.0, 0.0], :>=, 0.8)];
    name = :x_mid_or_high,
)
```

means:

```text
0.4 <= y[1] <= 0.6 OR y[1] >= 0.8
```

The same rule can be written with JuMP-style constraints:

```julia
using JuMP

y = output_variables(dm)

add_disjunction!(
    dm,
    [
        @build_constraint(y[1] >= 0.4),
        @build_constraint(y[1] <= 0.6),
    ],
    [@build_constraint(y[1] >= 0.8)];
    name = :x_mid_or_high,
)
```

The `name` keyword is optional, but recommended. Named rules can be used to control partial-DNF ordering:

```julia
rule_ordering = [:x_split, :y_split]
```

---

### `output_variables(model)`

Returns JuMP variables associated with the output coordinates. This is used with JuMP-style constraint construction.

```julia
using JuMP

y = output_variables(dm)

add_linear_constraint!(
    dm,
    @build_constraint(y[1] + y[2] >= 0.8),
)
```

This is often easier to read than manually writing coefficient vectors.

---

### `project(model, yhat; formulation = :dnf)`

Projects a raw prediction onto the selected convexified rule set.

```julia
yhat = [0.5, 0.1]

result = project(dm, yhat; formulation = :dnf)

println(result.status)
println(result.y)
```

The result contains:

```julia
result.y       # projected prediction
result.status  # solver termination status
result.model   # internal JuMP/DiffOpt model
```

Common choices:

```julia
project(dm, yhat; formulation = :dnf)

project(dm, yhat; formulation = :cnf)

project(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

---

### `DisjunctiveProjectionLayer(model; formulation = :dnf)`

Creates a differentiable projection layer.

```julia
layer = DisjunctiveProjectionLayer(dm; formulation = :partial_dnf)

y = layer(yhat)
```

The layer can be placed after a Flux neural network.

```julia
model = Chain(
    Dense(4 => 16, relu),
    Dense(16 => 2),
    layer,
)
```

---

### `constrained_model(backbone, model; kwargs...)`

Wraps a Flux neural network with a disjunctive projection layer.

```julia
model = constrained_model(
    backbone,
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
)
```

Calling the model evaluates:

```julia
yhat = backbone(x)
y = projection_layer(yhat)
```

Only the neural network backbone has trainable parameters. The projection layer is differentiable but has no trainable weights.

---

### `constrained_model(backbone, example_input) do dm ... end`

Creates a `DisjunctiveModel` automatically by inferring the output dimension from the neural network.

```julia
model = constrained_model(backbone, x0; formulation = :cnf) do dm
    set_bounds!(dm, lower = zeros(2), upper = ones(2))
    add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)
end
```

This is the most convenient API for Flux users.

---

## Projection API

Project a raw prediction:

```julia
yhat = [0.5, 0.1]

result = project(dm, yhat; formulation = :dnf)

println(result.status)
println(result.y)
```

Use a projection layer directly:

```julia
layer = DisjunctiveProjectionLayer(dm; formulation = :dnf)

y = layer(yhat)
```

Available formulations:

```julia
layer_dnf = DisjunctiveProjectionLayer(dm; formulation = :dnf)

layer_cnf = DisjunctiveProjectionLayer(dm; formulation = :cnf)

layer_partial = DisjunctiveProjectionLayer(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

---

## Full Flux example

The following example builds a small neural network, defines a disjunctive rule set, compares `:dnf`, `:cnf`, and `:partial_dnf` projections on the same raw neural network output, runs one training step, and performs inference on a single sample.

The same script is available under [`examples/flux_end2end.jl`](examples/flux_end2end.jl).

```julia
using Flux
using Zygote
using DisjunctiveDifferentiableLayers

# -----------------------------
# 1. Build a neural network
# -----------------------------

backbone = Chain(
    Dense(3 => 8, relu),
    Dense(8 => 2),
)

x = Float32[0.2, 0.7, 0.4]
target = Float32[0.8, 0.2]

# -----------------------------
# 2. Build the disjunctive rule model
# -----------------------------

dm = DisjunctiveModel(2)

set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

# Global constraint:
# y1 + y2 >= 0.8
add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

# Rule 1 has three disjuncts:
# y1 <= 0.2 OR 0.4 <= y1 <= 0.6 OR y1 >= 0.8
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.2)],
    [
        LinearConstraint([1.0, 0.0], :>=, 0.4),
        LinearConstraint([1.0, 0.0], :<=, 0.6),
    ],
    [LinearConstraint([1.0, 0.0], :>=, 0.8)];
    name = :x_rule,
)

# Rule 2 has three disjuncts:
# y2 <= 0.2 OR 0.35 <= y2 <= 0.55 OR y2 >= 0.7
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.2)],
    [
        LinearConstraint([0.0, 1.0], :>=, 0.35),
        LinearConstraint([0.0, 1.0], :<=, 0.55),
    ],
    [LinearConstraint([0.0, 1.0], :>=, 0.7)];
    name = :y_rule,
)

# -----------------------------
# 3. Inspect the model and formulations
# -----------------------------

println()
println("=== User-facing disjunctive model ===")
print_model(dm)

println()
println("=== DNF lifted formulation ===")
print_projection_model(dm; formulation = :dnf)

println()
println("=== CNF lifted formulation ===")
print_projection_model(dm; formulation = :cnf)

println()
println("=== Partial-DNF lifted formulation ===")
print_projection_model(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
)

# -----------------------------
# 4. Compare DNF, CNF, and partial-DNF on the same NN output
# -----------------------------

yhat = backbone(x)

println()
println("Raw NN output yhat = ", yhat)

dnf_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :dnf,
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

cnf_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :cnf,
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

partial_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

y_dnf = dnf_layer(yhat)
y_cnf = cnf_layer(yhat)
y_partial = partial_layer(yhat)

println()
println("=== Projected predictions for the same sample ===")
println("DNF projection         = ", y_dnf, "  sum = ", sum(y_dnf))
println("CNF projection         = ", y_cnf, "  sum = ", sum(y_cnf))
println("Partial-DNF projection = ", y_partial, "  sum = ", sum(y_partial))

println()
println("The three projections can differ because :dnf, :cnf, and :partial_dnf")
println("construct different convexified relaxations of the same logical rule set.")

# -----------------------------
# 5. Build a trainable constrained Flux model
# -----------------------------

# Here we train the partial-DNF constrained model.
# The projection layer is differentiable, but it has no trainable parameters.
# Flux only trains the neural network backbone.
model = constrained_model(
    backbone,
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_rule, :y_rule],
    y_regularization = 1e-4,
    ycopy_regularization = 1e-4,
    gamma_regularization = 1e-4,
    anchor_regularization = 1e-4,
)

# -----------------------------
# 6. Forward pass
# -----------------------------

y = model(x)

println()
println("=== Forward pass through constrained model ===")
println("Projected prediction = ", y)
println("Feasibility check: y1 + y2 = ", sum(y))

# -----------------------------
# 7. One training step
# -----------------------------

loss(m, x, target) = sum(abs2, m(x) .- target)

opt = Flux.setup(Adam(1e-3), model)

l, grads = Flux.withgradient(model) do m
    loss(m, x, target)
end

Flux.update!(opt, model, grads[1])

println()
println("Training loss before update = ", l)

# -----------------------------
# 8. Inference after one update
# -----------------------------

y_after = model(x)

println()
println("=== Inference after one update ===")
println("Projected prediction after one update = ", y_after)
println("Feasibility check after update: y1 + y2 = ", sum(y_after))
```

The constrained model is a Flux-compatible model.

---

## Formulation summaries and benchmarking

The package includes utilities for inspecting lifted formulations:

```julia
formulation_summary(dm; formulation = :dnf)

formulation_summary(dm; formulation = :cnf)

formulation_summary(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
```

Benchmark projection time and model size:

```julia
benchmark_projection(dm, yhat; formulation = :cnf, label = "CNF")

benchmark_projection(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
    label = "partial-DNF k=1",
)
```

Example output:

```text
CNF                  status=OPTIMAL build=0.002s solve=0.003s vars=54 cons=152 y=[0.5, 0.2]
partial-DNF k=1      status=OPTIMAL build=0.003s solve=0.004s vars=62 cons=180 y=[0.5, 0.2]
```

---

## Examples

Run examples from the package root:

```bash
julia --project=. examples/basic_example.jl
julia --project=. examples/midsize_example.jl
julia --project=. examples/largesize_example.jl
julia --project=. examples/flux_end2end.jl
julia --project=. scripts/verify_baseline.jl
julia --project=. scripts/verify_exact_oracle.jl
julia --project=. scripts/verify_lazy_search.jl
```

---

## Current limitations

- Constraints are currently linear.
- The differentiable backend relies on DiffOpt.
- Degenerate active sets use a minimum-norm LSQR sensitivity; the classical
  derivative need not be unique at such points.
- Full DNF can grow exponentially in the number of disjunctions.
- Partial-DNF is intended to trade off formulation strength and computational size.
- The package currently targets Flux.jl as the main neural network interface.

---

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{
anonymous2026disjunctivenet,
title={DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers},
author={Shraman Pal, Can Li},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=c88GPpURN8}
}
```

---

## License

This package is released under the MIT License. See [`LICENSE`](LICENSE) for details.
