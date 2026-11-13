"""Exact projection restricted to the center winner and certified competitor set."""
struct RelevantProjectionResult
    y::Vector{Float64}
    choices::Vector{Int}
    objective::Float64
    status::Symbol
    solved_branches::Int
    feasible_branches::Int
    infeasible_branches::Int
    unresolved_branches::Int
    winner_result::Union{Nothing, BranchProjectionResult}
end

function _visit_group_completions!(
    callback,
    model::DisjunctiveModel,
    assignments::Vector{Int},
    rule_index::Int = 1,
)
    if rule_index > length(assignments)
        callback(assignments)
        return nothing
    end
    if assignments[rule_index] != 0
        _visit_group_completions!(callback, model, assignments, rule_index + 1)
        return nothing
    end
    for choice in eachindex(model.disjunctions[rule_index].disjuncts)
        assignments[rule_index] = choice
        _visit_group_completions!(callback, model, assignments, rule_index + 1)
    end
    assignments[rule_index] = 0
    return nothing
end

"""
    project_relevant_branches(model, w, winner, groups; kwargs...)

Solve the fixed reference branch and every completion of the non-pruned,
disjoint competitor groups. Correctness requires the groups to come from a
switching certificate whose ball contains `w`.
"""
function project_relevant_branches(
    model::DisjunctiveModel,
    w::AbstractVector{<:Real},
    winner::AbstractVector{<:Integer},
    groups::AbstractVector{CompetitorGroup};
    reference_result = nothing,
    solver = nothing,
    feasibility_tol::Real = 1e-6,
    tie_tol::Real = 1e-10,
    max_relevant_branches::Int = 100_000,
)
    max_relevant_branches > 0 ||
        throw(ArgumentError("max_relevant_branches must be positive."))
    _validate_branch_choices(model, winner)
    relevant_count = 1 + sum(group.branch_count for group in groups; init = 0)
    relevant_count <= max_relevant_branches || throw(ArgumentError(
        "Relevant set contains $(relevant_count) branches, exceeding " *
        "max_relevant_branches=$(max_relevant_branches).",
    ))

    solved = 0
    feasible = 0
    infeasible = 0
    unresolved = 0
    best::Union{Nothing, BranchProjectionResult} = nothing

    function process!(branch::BranchProjectionResult)
        branch.solver_status == MOI.OPTIMAL && (solved += 1)
        if branch.accepted
            feasible += 1
            _lexicographically_better(branch, best, Float64(tie_tol)) && (best = branch)
        elseif branch.proven_infeasible
            infeasible += 1
        else
            unresolved += 1
        end
        return nothing
    end

    reference = reference_result === nothing ? project_branch(
        model,
        w,
        winner;
        solver = solver,
        feasibility_tol = feasibility_tol,
    ) : reference_result
    reference isa BranchProjectionResult ||
        throw(ArgumentError("reference_result must be a BranchProjectionResult."))
    reference.choices == Int.(collect(winner)) ||
        throw(ArgumentError("reference_result choices do not match winner."))
    process!(reference)

    for group in groups
        assignments = copy(group.assignments)
        _visit_group_completions!(model, assignments) do choices
            process!(project_branch(
                model,
                w,
                choices;
                solver = solver,
                feasibility_tol = feasibility_tol,
            ))
        end
    end

    status = unresolved > 0 ? :uncertified : best === nothing ? :infeasible : :optimal
    return RelevantProjectionResult(
        best === nothing ? Float64[] : copy(best.y),
        best === nothing ? Int[] : copy(best.choices),
        best === nothing ? Inf : best.objective,
        status,
        solved,
        feasible,
        infeasible,
        unresolved,
        best,
    )
end

