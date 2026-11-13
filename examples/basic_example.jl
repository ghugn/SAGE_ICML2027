using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DisjunctiveNet
using JuMP
import MathOptInterface as MOI

println("=== Basic disjunctive projection example ===")

dm = DisjunctiveModel(2)

set_bounds!(
    dm,
    lower = [0.0, 0.0],
    upper = [1.0, 1.0],
)

# Global constraint:
# y1 + y2 >= 0.8
add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

# Rule 1:
# y1 <= 0.25 OR y1 >= 0.75
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)];
    name = :x_split,
)

# Rule 2:
# y2 <= 0.25 OR y2 >= 0.75
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.25)],
    [LinearConstraint([0.0, 1.0], :>=, 0.75)];
    name = :y_split,
)

println()
println("=== User-facing model ===")
print_model(dm)

println()
println("=== Full DNF lifted model ===")
print_projection_model(dm; formulation = :dnf)

println()
println("=== CNF lifted model ===")
print_projection_model(dm; formulation = :cnf)

println()
println("=== Partial-DNF lifted model: first rule DNF, rest CNF ===")
print_projection_model(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)

yhat = [0.5, 0.1]

println()
println("Raw prediction yhat = ", yhat)

result_dnf = project(dm, yhat; formulation = :dnf)
result_cnf = project(dm, yhat; formulation = :cnf)
result_partial = project(
    dm,
    yhat;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)
result_exact = exhaustive_project(dm, yhat)
result_lazy = lazy_branch_search(dm, yhat)

println()
println("=== Projection results ===")
function print_verified_result(label, result)
    report = check_feasibility(dm, result.y)
    objective = result.status == MOI.OPTIMAL ? JuMP.objective_value(result.model) : NaN
    println(label, " status            = ", result.status)
    println(label, " y                 = ", result.y)
    println(label, " objective         = ", objective)
    println(label, " logical_feasible  = ", report.feasible)
    println(label, " max_violation     = ", report.max_violation)
end

print_verified_result("DNF", result_dnf)
print_verified_result("CNF", result_cnf)
print_verified_result("Partial", result_partial)

println("Exact status            = ", result_exact.status)
println("Exact branch choices    = ", result_exact.choices)
println("Exact y                 = ", result_exact.y)
println("Exact objective 0.5*d^2 = ", result_exact.objective)
println("Exact logical_feasible  = ", result_exact.report.feasible)
println("Exact max_violation     = ", result_exact.report.max_violation)

println("Lazy status             = ", result_lazy.status)
println("Lazy branch choices     = ", result_lazy.choices)
println("Lazy y                  = ", result_lazy.y)
println("Lazy objective 0.5*d^2  = ", result_lazy.objective)
println("Lazy leaf solves        = ", result_lazy.leaf_solves)
println("Lazy pruned branches    = ", result_lazy.pruned_branches)

println()
println("=== Layer API ===")

dnf_layer = DisjunctiveProjectionLayer(dm; formulation = :dnf)
cnf_layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)
partial_layer = DisjunctiveProjectionLayer(
    dm;
    formulation = :partial_dnf,
    num_dnf_rules = 1,
    rule_ordering = [:x_split, :y_split],
)

for (label, layer) in (
    ("dnf_layer", dnf_layer),
    ("cnf_layer", cnf_layer),
    ("partial_layer", partial_layer),
)
    y = layer(yhat)
    report = check_feasibility(dm, y)
    println(
        label,
        "(yhat) = ", y,
        " logical_feasible=", report.feasible,
        " max_violation=", report.max_violation,
    )
end
