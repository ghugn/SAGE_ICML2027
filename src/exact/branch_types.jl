"""
    BranchProjectionResult

Result of projecting onto one complete disjunct assignment.

`accepted` is true only when the continuous solver returned `OPTIMAL` and the
point passed original-logic verification. `proven_infeasible` distinguishes an
infeasibility certificate from an unresolved solver or numerical failure.
"""
struct BranchProjectionResult
    choices::Vector{Int}
    y::Vector{Float64}
    objective::Float64
    distance::Float64
    solver_status::Any
    report::Union{Nothing, FeasibilityReport}
    selected_branch_violation::Float64
    solve_time::Float64
    accepted::Bool
    proven_infeasible::Bool
    message::Union{Nothing, String}
end

"""
    ExactProjectionResult

Aggregate result of exhaustive branch projection.

`status` is one of:

- `:optimal`: every branch was resolved and `y` is globally exact;
- `:infeasible`: every branch was proved infeasible;
- `:uncertified`: at least one branch was unresolved, so the incumbent (if any)
  is feasible but is not a global certificate.
"""
struct ExactProjectionResult
    y::Vector{Float64}
    choices::Vector{Int}
    objective::Float64
    distance::Float64
    status::Symbol
    report::Union{Nothing, FeasibilityReport}
    total_branches::Int
    solved_branches::Int
    feasible_branches::Int
    infeasible_branches::Int
    unresolved_branches::Int
    branch_results::Vector{BranchProjectionResult}
end

"""Return whether an exhaustive result is a complete optimality/infeasibility certificate."""
is_certified(result::ExactProjectionResult) = result.status in (:optimal, :infeasible)

function Base.show(io::IO, result::ExactProjectionResult)
    print(
        io,
        "ExactProjectionResult(status=$(result.status), " *
        "objective=$(result.objective), choices=$(result.choices), " *
        "resolved=$(result.total_branches - result.unresolved_branches)/$(result.total_branches))",
    )
end

