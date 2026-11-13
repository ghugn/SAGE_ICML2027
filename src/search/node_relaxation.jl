function _validate_partial_assignments(
    model::DisjunctiveModel,
    assignments::AbstractVector{<:Integer},
)
    length(assignments) == length(model.disjunctions) ||
        throw(DimensionMismatch(
            "Expected $(length(model.disjunctions)) assignments, got $(length(assignments)).",
        ))
    for (rule_index, choice) in enumerate(assignments)
        n_disjuncts = length(model.disjunctions[rule_index].disjuncts)
        0 <= choice <= n_disjuncts ||
            throw(ArgumentError(
                "Assignment $(choice) for rule $(rule_index) is outside 0:$(n_disjuncts).",
            ))
    end
    return nothing
end

function _constraint_distance_lower_bound(
    constraint::LinearConstraint,
    yhat::AbstractVector{<:Real},
)
    residual = constraint_violation(constraint, yhat)
    residual == 0.0 && return 0.0
    norm_a = sqrt(sum(abs2, constraint.a))
    return norm_a == 0.0 ? Inf : residual / norm_a
end

"""
    geometric_lower_bound(model, yhat, assignments)

Analytic lower bound on Euclidean distance to every completion of a partial
assignment. Unassigned rules are encoded by zero.
"""
function geometric_lower_bound(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real},
    assignments::AbstractVector{<:Integer},
)
    length(yhat) == model.n_outputs ||
        throw(DimensionMismatch("yhat dimension does not match the model."))
    _validate_partial_assignments(model, assignments)

    box_distance_squared = 0.0
    for j in eachindex(yhat)
        delta =
            yhat[j] < model.lb[j] ? model.lb[j] - yhat[j] :
            yhat[j] > model.ub[j] ? yhat[j] - model.ub[j] : 0.0
        box_distance_squared += delta^2
    end
    distance_bound = sqrt(box_distance_squared)

    for constraint in model.constraints
        distance_bound = max(
            distance_bound,
            _constraint_distance_lower_bound(constraint, yhat),
        )
    end
    for (rule_index, choice) in enumerate(assignments)
        choice == 0 && continue
        for constraint in model.disjunctions[rule_index].disjuncts[choice]
            distance_bound = max(
                distance_bound,
                _constraint_distance_lower_bound(constraint, yhat),
            )
        end
    end

    return 0.5 * distance_bound^2
end

function _partial_constraint_violation(
    model::DisjunctiveModel,
    y::AbstractVector{<:Real},
    assignments::AbstractVector{<:Integer},
)
    violation = 0.0
    for j in eachindex(y)
        isfinite(model.lb[j]) && (violation = max(violation, model.lb[j] - y[j]))
        isfinite(model.ub[j]) && (violation = max(violation, y[j] - model.ub[j]))
    end
    for constraint in model.constraints
        violation = max(violation, constraint_violation(constraint, y))
    end
    for (rule_index, choice) in enumerate(assignments)
        choice == 0 && continue
        for constraint in model.disjunctions[rule_index].disjuncts[choice]
            violation = max(violation, constraint_violation(constraint, y))
        end
    end
    return max(violation, 0.0)
end

"""Solve the continuous QP for a partial assignment; zero means unassigned."""
function solve_node_relaxation(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real},
    assignments::AbstractVector{<:Integer};
    solver = nothing,
    feasibility_tol::Real = 1e-6,
)
    length(yhat) == model.n_outputs ||
        throw(DimensionMismatch("yhat dimension does not match the model."))
    all(isfinite, yhat) || throw(ArgumentError("yhat must contain only finite values."))
    feasibility_tol >= 0 || throw(ArgumentError("feasibility_tol must be nonnegative."))
    _validate_partial_assignments(model, assignments)

    assignment_vector = Int.(collect(assignments))
    certified_bound = geometric_lower_bound(model, yhat, assignment_vector)
    jump_model, y = _build_selected_branch_model(
        model,
        yhat,
        assignment_vector;
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
        return NodeRelaxationResult(
            Float64[], Inf, certified_bound, :solver_error, :unresolved,
            Inf, solve_time, solve_error[],
        )
    end

    solver_status = termination_status(jump_model)
    if solver_status == MOI.INFEASIBLE
        return NodeRelaxationResult(
            Float64[], Inf, Inf, solver_status, :infeasible,
            0.0, solve_time, nothing,
        )
    elseif solver_status != MOI.OPTIMAL
        return NodeRelaxationResult(
            Float64[], Inf, certified_bound, solver_status, :unresolved,
            Inf, solve_time, "Node relaxation did not return OPTIMAL or INFEASIBLE.",
        )
    end

    projected = Float64.(value.(y))
    primal_objective = 0.5 * sum(abs2, projected .- Float64.(yhat))
    violation = _partial_constraint_violation(model, projected, assignment_vector)
    if !isfinite(primal_objective) || violation > feasibility_tol
        return NodeRelaxationResult(
            projected, primal_objective, certified_bound, solver_status, :unresolved,
            violation, solve_time,
            "Optimal node relaxation failed its partial-constraint residual check.",
        )
    end

    return NodeRelaxationResult(
        projected,
        primal_objective,
        certified_bound,
        solver_status,
        :optimal,
        violation,
        solve_time,
        nothing,
    )
end

