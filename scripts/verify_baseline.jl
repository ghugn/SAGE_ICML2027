using DisjunctiveNet
using JuMP
using Zygote
import MathOptInterface as MOI

println("DisjunctiveNet Phase 0 baseline verification")
println("Julia version: ", VERSION)

dm = DisjunctiveModel(2)
set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])
add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)
add_disjunction!(
    dm,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)];
    name = :x_split,
)
add_disjunction!(
    dm,
    [LinearConstraint([0.0, 1.0], :<=, 0.25)],
    [LinearConstraint([0.0, 1.0], :>=, 0.75)];
    name = :y_split,
)

yhat = [0.5, 0.1]
settings = (
    (:dnf, -1, nothing),
    (:cnf, -1, nothing),
    (:partial_dnf, 1, [:x_split, :y_split]),
)

println("\nformulation | status | objective | logical_feasible | max_violation | y")
for (formulation, num_dnf_rules, rule_ordering) in settings
    result = project(
        dm,
        yhat;
        formulation = formulation,
        num_dnf_rules = num_dnf_rules,
        rule_ordering = rule_ordering,
    )
    report = check_feasibility(dm, result.y)
    objective = result.status == MOI.OPTIMAL ? JuMP.objective_value(result.model) : NaN
    println(
        formulation, " | ", result.status, " | ", objective, " | ",
        report.feasible, " | ", report.max_violation, " | ", result.y,
    )
    @assert result.status == MOI.OPTIMAL
    @assert isfinite(objective)
    @assert isfinite(report.max_violation)
end

# Deliberately degenerate active set: verifies the robust reverse KKT fallback.
stress = DisjunctiveModel(2)
set_bounds!(stress, lower = [0.0, 0.0], upper = [1.0, 1.0])
add_linear_constraint!(stress, [1.0, 1.0], :(==), 1.0)
add_linear_constraint!(stress, [2.0, 2.0], :(==), 2.0)
add_disjunction!(
    stress,
    [LinearConstraint([1.0, 0.0], :<=, 0.25)],
    [LinearConstraint([1.0, 0.0], :>=, 0.75)],
)
add_disjunction!(
    stress,
    [LinearConstraint([0.0, 1.0], :<=, 0.25)],
    [LinearConstraint([0.0, 1.0], :>=, 0.75)],
)

stress_layer = DisjunctiveProjectionLayer(stress)
gradient = Zygote.gradient(z -> sum(stress_layer(z)), [0.5, 0.5])[1]
@assert all(isfinite, gradient)
println("\nDegenerate KKT gradient: ", gradient)
println("BASELINE_VERIFICATION=PASS")

