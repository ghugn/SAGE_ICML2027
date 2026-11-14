using JuMP
using HiGHS
using LinearAlgebra
import MathOptInterface as MOI

"""
    num_branches(model)

Return the Cartesian-product size of the model's disjunct choices. A model with
no disjunctions has one ordinary convex branch.
"""
function num_branches(model::DisjunctiveModel)
    count = 1
    for rule in model.disjunctions
        count = Base.checked_mul(count, length(rule.disjuncts))
    end
    return count
end

function _validate_branch_choices(model::DisjunctiveModel, choices::AbstractVector{<:Integer})
    length(choices) == length(model.disjunctions) ||
        throw(DimensionMismatch(
            "Expected $(length(model.disjunctions)) branch choices, got $(length(choices)).",
        ))

    for (rule_index, choice) in enumerate(choices)
        n_disjuncts = length(model.disjunctions[rule_index].disjuncts)
        1 <= choice <= n_disjuncts ||
            throw(ArgumentError(
                "Choice $(choice) for rule $(rule_index) is outside 1:$(n_disjuncts).",
            ))
    end
    return nothing
end

function _add_exact_constraint!(jump_model::JuMP.Model, constraint::LinearConstraint, y)
    lhs = sum(constraint.a[j] * y[j] for j in eachindex(constraint.a))
    if constraint.sense == :<=
        @constraint(jump_model, lhs <= constraint.b)
    elseif constraint.sense == :>=
        @constraint(jump_model, lhs >= constraint.b)
    else
        @constraint(jump_model, lhs == constraint.b)
    end
    return nothing
end

function _selected_branch_violation(
    model::DisjunctiveModel,
    y::AbstractVector{<:Real},
    choices::AbstractVector{<:Integer},
)
    violation = 0.0
    for (rule_index, choice) in enumerate(choices)
        for constraint in model.disjunctions[rule_index].disjuncts[choice]
            violation = max(violation, constraint_violation(constraint, y))
        end
    end
    return violation
end

function _build_selected_branch_model(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real},
    assignments::AbstractVector{<:Integer};
    solver = nothing,
)
    optimizer = solver === nothing ? HiGHS.Optimizer : solver
    jump_model = Model(optimizer)
    set_silent(jump_model)

    @variable(jump_model, y[1:model.n_outputs])
    for j in eachindex(y)
        if isfinite(model.lb[j])
            set_lower_bound(y[j], model.lb[j])
        end
        if isfinite(model.ub[j])
            set_upper_bound(y[j], model.ub[j])
        end
    end

    for constraint in model.constraints
        _add_exact_constraint!(jump_model, constraint, y)
    end
    for (rule_index, choice) in enumerate(assignments)
        choice == 0 && continue
        for constraint in model.disjunctions[rule_index].disjuncts[choice]
            _add_exact_constraint!(jump_model, constraint, y)
        end
    end

    @objective(
        jump_model,
        Min,
        0.5 * sum((y[j] - Float64(yhat[j]))^2 for j in eachindex(y)),
    )
    return jump_model, y
end

function _analytic_branch_projection(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real},
    assignments::AbstractVector{<:Integer};
    feasibility_tol::Real = 1e-6,
    max_active_sets::Int = 100_000,
)
    n = model.n_outputs
    inequality_rows = Vector{Vector{Float64}}()
    inequality_bounds = Float64[]
    equality_rows = Vector{Vector{Float64}}()
    equality_bounds = Float64[]

    function add_constraint!(constraint::LinearConstraint)
        row = Float64.(constraint.a)
        if constraint.sense == :<=
            push!(inequality_rows, row)
            push!(inequality_bounds, constraint.b)
        elseif constraint.sense == :>=
            push!(inequality_rows, -row)
            push!(inequality_bounds, -constraint.b)
        else
            push!(equality_rows, row)
            push!(equality_bounds, constraint.b)
        end
        return nothing
    end

    for coordinate in 1:n
        if isfinite(model.lb[coordinate])
            row = zeros(n)
            row[coordinate] = -1.0
            push!(inequality_rows, row)
            push!(inequality_bounds, -model.lb[coordinate])
        end
        if isfinite(model.ub[coordinate])
            row = zeros(n)
            row[coordinate] = 1.0
            push!(inequality_rows, row)
            push!(inequality_bounds, model.ub[coordinate])
        end
    end
    foreach(add_constraint!, model.constraints)
    for (rule_index, choice) in enumerate(assignments)
        choice == 0 && continue
        foreach(add_constraint!, model.disjunctions[rule_index].disjuncts[choice])
    end

    inequalities = isempty(inequality_rows) ? zeros(0, n) :
        reduce(vcat, permutedims.(inequality_rows))
    equalities = isempty(equality_rows) ? zeros(0, n) :
        reduce(vcat, permutedims.(equality_rows))
    point = Float64.(yhat)
    tolerance = max(Float64(feasibility_tol), 1e-9)
    best = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    best_objective = Ref(Inf)
    visited = Ref(0)

    function consider!(active::Vector{Int})
        visited[] += 1
        visited[] <= max_active_sets || return nothing
        rows = isempty(active) ? equalities : vcat(equalities, inequalities[active, :])
        bounds = isempty(active) ? equality_bounds :
            vcat(equality_bounds, inequality_bounds[active])
        size(rows, 1) <= n || return nothing
        rank(rows; atol = 1e-10) == size(rows, 1) || return nothing
        multipliers = isempty(bounds) ? Float64[] :
            (rows * transpose(rows)) \ (rows * point - bounds)
        if !isempty(active)
            all(multipliers[length(equality_bounds) + 1:end] .>= -tolerance) ||
                return nothing
        end
        candidate = point - transpose(rows) * multipliers
        all(isfinite, candidate) || return nothing
        all(inequalities * candidate .<= inequality_bounds .+ tolerance) ||
            return nothing
        all(abs.(equalities * candidate .- equality_bounds) .<= tolerance) ||
            return nothing
        objective = 0.5 * sum(abs2, candidate .- point)
        if objective < best_objective[]
            best[] = candidate
            best_objective[] = objective
        end
        return nothing
    end

    chosen = Int[]
    function enumerate!(first_index::Int, remaining::Int)
        visited[] >= max_active_sets && return nothing
        if remaining == 0
            consider!(chosen)
            return nothing
        end
        last_index = size(inequalities, 1) - remaining + 1
        for index in first_index:last_index
            push!(chosen, index)
            enumerate!(index + 1, remaining - 1)
            pop!(chosen)
            visited[] >= max_active_sets && break
        end
        return nothing
    end

    maximum_active = min(n - size(equalities, 1), size(inequalities, 1))
    maximum_active < 0 && return nothing
    for count in 0:maximum_active
        enumerate!(1, count)
        visited[] >= max_active_sets && break
    end
    return best[]
end

"""
    project_branch(model, yhat, choices; solver=nothing, feasibility_tol=1e-6)

Project `yhat` onto one selected original branch. No convex-hull variables or
logical relaxation are introduced.
"""
function project_branch(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real},
    choices::AbstractVector{<:Integer};
    solver = nothing,
    feasibility_tol::Real = 1e-6,
)
    length(yhat) == model.n_outputs ||
        throw(DimensionMismatch(
            "Expected yhat of length $(model.n_outputs), got $(length(yhat)).",
        ))
    all(isfinite, yhat) || throw(ArgumentError("yhat must contain only finite values."))
    feasibility_tol >= 0 || throw(ArgumentError("feasibility_tol must be nonnegative."))
    _validate_branch_choices(model, choices)

    branch_choices = Int.(collect(choices))
    jump_model, y = _build_selected_branch_model(
        model,
        yhat,
        branch_choices;
        solver = solver,
    )

    solve_error = Ref{Union{Nothing, String}}(nothing)
    solve_time = @elapsed begin
        try
            optimize!(jump_model)
        catch err
            solve_error[] = sprint(showerror, err)
        end
    end

    if solve_error[] !== nothing
        return BranchProjectionResult(
            branch_choices,
            Float64[],
            Inf,
            Inf,
            :solver_error,
            nothing,
            Inf,
            solve_time,
            false,
            false,
            solve_error[],
        )
    end

    solver_status = termination_status(jump_model)
    if solver_status == MOI.INFEASIBLE
        return BranchProjectionResult(
            branch_choices,
            Float64[],
            Inf,
            Inf,
            solver_status,
            nothing,
            0.0,
            solve_time,
            false,
            true,
            nothing,
        )
    elseif solver_status != MOI.OPTIMAL
        # HiGHS can return OTHER_ERROR for a nearly degenerate convex QP after
        # its candidate fails the final primal-feasibility check. Retry the
        # identical feasible set as an LP before declaring the branch
        # unresolved. Only an explicit LP INFEASIBLE status is used as a
        # certificate; an LP-feasible branch still needs an exact projection.
        fallback_started = time_ns()
        feasibility_status = try
            @objective(jump_model, Min, 0.0)
            optimize!(jump_model)
            termination_status(jump_model)
        catch
            :solver_error
        end
        if feasibility_status == MOI.INFEASIBLE
            return BranchProjectionResult(
                branch_choices,
                Float64[],
                Inf,
                Inf,
                feasibility_status,
                nothing,
                0.0,
                solve_time + (time_ns() - fallback_started) / 1e9,
                false,
                true,
                "QP status $(solver_status); LP feasibility retry certified infeasibility.",
            )
        end
        if feasibility_status == MOI.OPTIMAL
            projected = _analytic_branch_projection(
                model,
                yhat,
                branch_choices;
                feasibility_tol = feasibility_tol,
            )
            if projected !== nothing
                objective = 0.5 * sum(abs2, projected .- Float64.(yhat))
                report = check_feasibility(model, projected; atol = feasibility_tol)
                branch_violation = _selected_branch_violation(
                    model, projected, branch_choices,
                )
                accepted = report.feasible &&
                    branch_violation <= feasibility_tol && isfinite(objective)
                if accepted
                    return BranchProjectionResult(
                        branch_choices,
                        projected,
                        objective,
                        sqrt(2 * objective),
                        MOI.OPTIMAL,
                        report,
                        branch_violation,
                        solve_time + (time_ns() - fallback_started) / 1e9,
                        true,
                        false,
                        "QP status $(solver_status); resolved by analytic active-set fallback.",
                    )
                end
            end
        end
        return BranchProjectionResult(
            branch_choices,
            Float64[],
            Inf,
            Inf,
            solver_status,
            nothing,
            Inf,
            solve_time + (time_ns() - fallback_started) / 1e9,
            false,
            false,
            "Branch QP returned $(solver_status); LP feasibility retry returned " *
            "$(feasibility_status).",
        )
    end

    projected = Float64.(value.(y))
    objective = 0.5 * sum(abs2, projected .- Float64.(yhat))
    report = check_feasibility(model, projected; atol = feasibility_tol)
    branch_violation = _selected_branch_violation(model, projected, branch_choices)
    accepted =
        report.feasible &&
        branch_violation <= feasibility_tol &&
        isfinite(objective)
    message = accepted ? nothing :
        "OPTIMAL branch output failed original-logic verification."

    return BranchProjectionResult(
        branch_choices,
        projected,
        objective,
        sqrt(2 * objective),
        solver_status,
        report,
        branch_violation,
        solve_time,
        accepted,
        false,
        message,
    )
end

