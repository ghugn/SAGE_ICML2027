using LinearAlgebra

"""Diagnostics for the almost-everywhere fixed-branch implicit gradient."""
struct FixedBranchGradientResult
    gradient::Vector{Float64}
    jacobian::Matrix{Float64}
    y::Vector{Float64}
    active_constraints::Int
    active_rank::Int
    status::Symbol
    branch_result::BranchProjectionResult
    message::Union{Nothing, String}
end

function _push_active_constraint!(
    rows::Vector{Vector{Float64}},
    constraint::LinearConstraint,
    y::Vector{Float64},
    active_tol::Float64,
)
    value = dot(constraint.a, y)
    if constraint.sense == :(==)
        push!(rows, copy(constraint.a))
    elseif constraint.sense == :<=
        constraint.b - value <= active_tol && push!(rows, copy(constraint.a))
    else
        value - constraint.b <= active_tol && push!(rows, -copy(constraint.a))
    end
    return nothing
end

function _active_branch_matrix(
    model::DisjunctiveModel,
    y::Vector{Float64},
    choices::Vector{Int},
    active_tol::Float64,
)
    rows = Vector{Float64}[]
    dimension = model.n_outputs

    for coordinate in 1:dimension
        if isfinite(model.lb[coordinate]) && y[coordinate] - model.lb[coordinate] <= active_tol
            row = zeros(dimension)
            row[coordinate] = -1.0
            push!(rows, row)
        end
        if isfinite(model.ub[coordinate]) && model.ub[coordinate] - y[coordinate] <= active_tol
            row = zeros(dimension)
            row[coordinate] = 1.0
            push!(rows, row)
        end
    end
    for constraint in model.constraints
        _push_active_constraint!(rows, constraint, y, active_tol)
    end
    for (rule_index, choice) in enumerate(choices)
        for constraint in model.disjunctions[rule_index].disjuncts[choice]
            _push_active_constraint!(rows, constraint, y, active_tol)
        end
    end

    isempty(rows) && return zeros(0, dimension)
    return reduce(vcat, transpose.(rows))
end

"""
    fixed_branch_loss_gradient(model, w, choices, loss_gradient; kwargs...)

Compute the almost-everywhere implicit gradient of
`loss(project_branch(choices, w))` for a polyhedral branch. On an unchanged
active face, the projection Jacobian is
`I - A' * pinv(A*A') * A`, where `A` contains active constraint normals.
"""
function fixed_branch_loss_gradient(
    model::DisjunctiveModel,
    w::AbstractVector{<:Real},
    choices::AbstractVector{<:Integer},
    loss_gradient;
    branch_result = nothing,
    solver = nothing,
    feasibility_tol::Real = 1e-6,
    active_tol::Real = 1e-6,
    rank_tol::Real = 1e-10,
)
    active_tol >= 0 || throw(ArgumentError("active_tol must be nonnegative."))
    rank_tol >= 0 || throw(ArgumentError("rank_tol must be nonnegative."))
    point = Float64.(collect(w))
    branch_choices = Int.(collect(choices))
    result = branch_result === nothing ? project_branch(
        model,
        point,
        branch_choices;
        solver = solver,
        feasibility_tol = feasibility_tol,
    ) : branch_result

    result isa BranchProjectionResult ||
        throw(ArgumentError("branch_result must be a BranchProjectionResult."))
    result.choices == branch_choices ||
        throw(ArgumentError("branch_result choices do not match choices."))
    if !result.accepted
        return FixedBranchGradientResult(
            Float64[], zeros(0, 0), Float64[], 0, 0, :unresolved,
            result, "The fixed branch projection was not accepted.",
        )
    end

    y = copy(result.y)
    output_gradient = Float64.(collect(loss_gradient(y)))
    length(output_gradient) == model.n_outputs ||
        throw(DimensionMismatch("loss_gradient must return one value per output."))
    all(isfinite, output_gradient) ||
        throw(ArgumentError("loss_gradient returned a non-finite value."))

    active = _active_branch_matrix(
        model,
        y,
        branch_choices,
        Float64(active_tol),
    )
    dimension = model.n_outputs
    jacobian = Matrix{Float64}(I, dimension, dimension)
    active_rank = 0
    if size(active, 1) > 0
        active_rank = rank(active; atol = Float64(rank_tol), rtol = Float64(rank_tol))
        jacobian .-= active' * pinv(
            active * active';
            atol = Float64(rank_tol),
            rtol = Float64(rank_tol),
        ) * active
        jacobian .= 0.5 .* (jacobian .+ jacobian')
    end

    return FixedBranchGradientResult(
        jacobian' * output_gradient,
        jacobian,
        y,
        size(active, 1),
        active_rank,
        :ok,
        result,
        nothing,
    )
end

