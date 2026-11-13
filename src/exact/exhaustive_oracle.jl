"""
    exhaustive_project(model, yhat; kwargs...)

Enumerate complete branch assignments lazily, solve every branch projection,
and return a globally exact result when all branches are resolved.

This function is a small-instance oracle. `max_branches` prevents accidental
use on a combinatorially large model; raise it explicitly only for controlled
experiments.
"""
function exhaustive_project(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    feasibility_tol::Real = 1e-6,
    tie_tol::Real = 1e-10,
    max_branches::Int = 100_000,
    store_branch_results::Bool = true,
)
    tie_tol >= 0 || throw(ArgumentError("tie_tol must be nonnegative."))
    max_branches > 0 || throw(ArgumentError("max_branches must be positive."))

    total = num_branches(model)
    total <= max_branches ||
        throw(ArgumentError(
            "Exact oracle requires $(total) branches, exceeding max_branches=$(max_branches). " *
            "Increase the guard explicitly only for a controlled experiment.",
        ))

    stored = BranchProjectionResult[]
    solved = 0
    feasible = 0
    infeasible = 0
    unresolved = 0
    best::Union{Nothing, BranchProjectionResult} = nothing

    function process_choice!(choices)
        branch = project_branch(
            model,
            yhat,
            choices;
            solver = solver,
            feasibility_tol = feasibility_tol,
        )
        store_branch_results && push!(stored, branch)

        if branch.solver_status == MOI.OPTIMAL
            solved += 1
        end
        if branch.accepted
            feasible += 1
            if best === nothing ||
               branch.objective < best.objective - tie_tol ||
               (abs(branch.objective - best.objective) <= tie_tol &&
                isless(Tuple(branch.choices), Tuple(best.choices)))
                best = branch
            end
        elseif branch.proven_infeasible
            infeasible += 1
        else
            unresolved += 1
        end
        return nothing
    end

    if isempty(model.disjunctions)
        process_choice!(Int[])
    else
        ranges = [Base.OneTo(length(rule.disjuncts)) for rule in model.disjunctions]
        for choice_tuple in Iterators.product(ranges...)
            process_choice!(Int[choice_tuple...])
        end
    end

    if unresolved > 0
        status = :uncertified
    elseif best === nothing
        status = :infeasible
    else
        status = :optimal
    end

    if best === nothing
        return ExactProjectionResult(
            Float64[],
            Int[],
            Inf,
            Inf,
            status,
            nothing,
            total,
            solved,
            feasible,
            infeasible,
            unresolved,
            stored,
        )
    end

    return ExactProjectionResult(
        copy(best.y),
        copy(best.choices),
        best.objective,
        best.distance,
        status,
        best.report,
        total,
        solved,
        feasible,
        infeasible,
        unresolved,
        stored,
    )
end

"""Alias emphasizing that `exhaustive_project` is the exact small-instance projector."""
exact_project(args...; kwargs...) = exhaustive_project(args...; kwargs...)

