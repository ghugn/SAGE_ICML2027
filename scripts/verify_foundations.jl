using DisjunctiveNet
using JuMP
using Random
using Zygote
import MathOptInterface as MOI

function counterexample_model()
    model = DisjunctiveModel(2)
    set_bounds!(model, lower = [0.0, 0.0], upper = [1.0, 1.0])
    add_linear_constraint!(model, [1.0, 1.0], :>=, 0.8)
    add_disjunction!(
        model,
        [LinearConstraint([1.0, 0.0], :<=, 0.25)],
        [LinearConstraint([1.0, 0.0], :>=, 0.75)];
        name = :x_split,
    )
    add_disjunction!(
        model,
        [LinearConstraint([0.0, 1.0], :<=, 0.25)],
        [LinearConstraint([0.0, 1.0], :>=, 0.75)];
        name = :y_split,
    )
    return model
end

function verify_baseline_and_logic()
    model = counterexample_model()
    yhat = [0.5, 0.1]
    settings = (
        (:dnf, -1, nothing),
        (:cnf, -1, nothing),
        (:partial_dnf, 1, [:x_split, :y_split]),
    )
    reports = Dict{Symbol, FeasibilityReport}()

    println("\n[Gate 1] Reproducible convexified baseline")
    println("formulation | status | finite_objective | logical_feasible | max_violation")
    for (formulation, num_dnf_rules, rule_ordering) in settings
        result = project(
            model,
            yhat;
            formulation = formulation,
            num_dnf_rules = num_dnf_rules,
            rule_ordering = rule_ordering,
        )
        report = check_feasibility(model, result.y)
        objective = result.status == MOI.OPTIMAL ? objective_value(result.model) : NaN
        reports[formulation] = report
        println(
            formulation, " | ", result.status, " | ", isfinite(objective),
            " | ", report.feasible, " | ", report.max_violation,
        )
        @assert result.status == MOI.OPTIMAL
        @assert isfinite(objective)
        @assert all(isfinite, result.y)
    end

    # The controlled gap point must expose relaxation/original-logic mismatch.
    @assert !reports[:dnf].feasible
    @assert !reports[:cnf].feasible
    @assert !reports[:partial_dnf].feasible

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
    gradient = Zygote.gradient(
        z -> sum(DisjunctiveProjectionLayer(stress)(z)),
        [0.5, 0.5],
    )[1]
    @assert all(isfinite, gradient)
    println("finite degenerate-KKT gradient: ", gradient)

    println("\n[Gate 2] Original-logic validator")
    feasible = check_feasibility(model, [0.75, 0.1])
    gap = check_feasibility(model, [0.5, 0.3])
    nonfinite = check_feasibility(model, [NaN, 0.0])
    @assert feasible.feasible
    @assert feasible.max_violation == 0.0
    @assert !gap.feasible && gap.max_violation > 0.0
    @assert !nonfinite.feasible && nonfinite.max_violation == Inf
    println("known feasible point accepted: ", feasible.feasible)
    println("logical-gap point rejected: ", !gap.feasible)
    println("non-finite point rejected: ", !nonfinite.feasible)
    return model, yhat
end

function verify_exact_oracle(model, yhat)
    println("\n[Gate 3] Exact branch-wise ground-truth oracle")
    exact = exhaustive_project(model, yhat)
    @assert exact.status == :optimal
    @assert is_certified(exact)
    @assert exact.choices == [2, 1]
    @assert isapprox(exact.y, [0.75, 0.1]; atol = 1e-7)
    @assert exact.report.feasible
    @assert exact.unresolved_branches == 0
    @assert exact.feasible_branches + exact.infeasible_branches == exact.total_branches
    println("counterexample winner/status: ", exact.choices, "/", exact.status)
    println("counterexample exact y/objective: ", exact.y, "/", exact.objective)
    println("resolved branches: ", exact.total_branches - exact.unresolved_branches,
            "/", exact.total_branches)

    interval = DisjunctiveModel(1)
    add_disjunction!(
        interval,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)],
    )
    tie = exhaustive_project(interval, [3.5])
    @assert tie.status == :optimal
    @assert tie.choices == [1]
    @assert isapprox(tie.y, [2.0]; atol = 1e-7)
    @assert isapprox(tie.objective, 1.125; atol = 1e-8)
    println("deterministic analytic tie winner: ", tie.choices)

    rng = MersenneTwister(20261113)
    random_cases = 0
    for dimension in (2, 3)
        separated = DisjunctiveModel(dimension)
        set_bounds!(
            separated;
            lower = fill(-2.0, dimension),
            upper = fill(2.0, dimension),
        )
        for coordinate in 1:dimension
            direction = zeros(dimension)
            direction[coordinate] = 1.0
            add_disjunction!(
                separated,
                [LinearConstraint(direction, :<=, -0.4)],
                [LinearConstraint(direction, :>=, 0.6)],
            )
        end
        for _ in 1:12
            point = -1.5 .+ 3.0 .* rand(rng, dimension)
            expected = map(point) do value
                value <= -0.4 || value >= 0.6 ? value :
                value <= 0.1 ? -0.4 : 0.6
            end
            result = exhaustive_project(separated, point)
            @assert result.status == :optimal
            @assert result.report.feasible
            @assert isapprox(result.y, expected; atol = 1e-6)
            @assert isapprox(
                result.objective,
                0.5 * sum(abs2, expected .- point);
                atol = 1e-7,
            )
            random_cases += 1
        end
    end
    println("independent analytic random cases: ", random_cases, "/", random_cases)
    return exact
end

function main()
    println("DisjunctiveNet pre-SAGE foundation verification")
    println("Julia version: ", VERSION)
    model, yhat = verify_baseline_and_logic()
    exact = verify_exact_oracle(model, yhat)
    println("\nFoundation exact output logically feasible: ", exact.report.feasible)
    println("FOUNDATION_VERIFICATION=PASS")
end

main()

