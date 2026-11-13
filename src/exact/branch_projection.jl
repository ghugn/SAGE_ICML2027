using JuMP
using HiGHS
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
        return BranchProjectionResult(
            branch_choices,
            Float64[],
            Inf,
            Inf,
            solver_status,
            nothing,
            Inf,
            solve_time,
            false,
            false,
            "Branch solver did not return OPTIMAL or INFEASIBLE.",
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

