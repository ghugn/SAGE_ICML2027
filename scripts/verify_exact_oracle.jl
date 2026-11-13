using DisjunctiveNet

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

exact_ref = Ref{ExactProjectionResult}()
elapsed = @elapsed exact_ref[] = exhaustive_project(dm, yhat)
exact = exact_ref[]
relaxed = project(dm, yhat; formulation = :dnf)
relaxed_report = check_feasibility(dm, relaxed.y)

println("DisjunctiveNet Phase 1 exact-oracle verification")
println("Julia version: ", VERSION)
println("Input yhat: ", yhat)
println("Total branches: ", exact.total_branches)
println("Exact status: ", exact.status)
println("Certified: ", is_certified(exact))
println("Winning choices: ", exact.choices)
println("Exact y: ", exact.y)
println("Exact objective (0.5*d^2): ", exact.objective)
println("Exact max logical violation: ", exact.report.max_violation)
println("Feasible/infeasible/unresolved branches: ", (
    exact.feasible_branches,
    exact.infeasible_branches,
    exact.unresolved_branches,
))
println("Wall time: ", elapsed, " seconds")
println()
println("Relaxed DNF y: ", relaxed.y)
println("Relaxed DNF logically feasible: ", relaxed_report.feasible)
println("Relaxed DNF max logical violation: ", relaxed_report.max_violation)

@assert exact.status == :optimal
@assert is_certified(exact)
@assert exact.choices == [2, 1]
@assert isapprox(exact.y, [0.75, 0.1]; atol = 1e-7)
@assert exact.report.feasible
@assert exact.unresolved_branches == 0
@assert !relaxed_report.feasible

println("EXACT_ORACLE_VERIFICATION=PASS")

