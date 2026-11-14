using Random
using LinearAlgebra

"""Exact proposal samples and importance weights for geometry-aware SAGE."""
struct GeometrySampleBatch
    samples::Matrix{Float64}
    importance_weights::Vector{Float64}
    proposal_ratios::Vector{Float64}
    components::Vector{Symbol}
    normals::Matrix{Float64}
    mixture_weight::Float64
    focus_strength::Float64
    beta::Float64
end

"""Optimizer-derived switching normals and the branch-solve cost used to obtain them."""
struct SwitchingGeometry
    normals::Matrix{Float64}
    competitor_choices::Vector{Vector{Int}}
    branch_solves::Int
    accepted_branches::Int
    unresolved_branches::Int
    degenerate_branches::Int
    status::Symbol
end

function _normalized_geometry_normals(normals, dimension::Int; tolerance::Float64 = 1e-10)
    matrix = Matrix{Float64}(normals)
    size(matrix, 1) == dimension || throw(DimensionMismatch(
        "geometry normals must have one row per model output.",
    ))
    size(matrix, 2) > 0 || throw(ArgumentError("geometry normals cannot be empty."))
    all(isfinite, matrix) || throw(ArgumentError("geometry normals must be finite."))
    tolerance >= 0 || throw(ArgumentError("normal tolerance must be nonnegative."))

    normalized = similar(matrix)
    for column in axes(matrix, 2)
        magnitude = norm(view(matrix, :, column))
        magnitude > tolerance || throw(ArgumentError(
            "every geometry normal must have norm above the requested tolerance.",
        ))
        normalized[:, column] .= view(matrix, :, column) ./ magnitude
    end
    return normalized
end

"""
    geometry_proposal_ratio(u, normals; beta=2, mixture_weight=0.5, focus_strength=0.9)

Return `p_z(u) / q(u)` for the proposal-aligned mixture

`p_z = (1 - alpha) q + alpha mean_j p_j`,

where

`p_j(u) / q(u) = (1 + kappa * dot(n_j, u))^2 / Z`,

`Z = 1 + kappa^2 / (d + 2beta + 2)`. Every `n_j` is normalized. The
normalizer follows from isotropy of the compact polynomial-ball density. With
`alpha < 1`, the proposal has support everywhere that `q` is positive.
"""
function geometry_proposal_ratio(
    u::AbstractVector{<:Real},
    normals;
    beta::Real = 2.0,
    mixture_weight::Real = 0.5,
    focus_strength::Real = 0.9,
)
    point = Float64.(collect(u))
    dimension = length(point)
    dimension > 0 || throw(ArgumentError("u cannot be empty."))
    all(isfinite, point) || throw(ArgumentError("u must be finite."))
    sum(abs2, point) < 1.0 || throw(ArgumentError(
        "geometry proposal density is defined only inside the open unit ball.",
    ))
    isfinite(beta) && beta >= 2.0 || throw(ArgumentError(
        "beta must be finite and at least 2.",
    ))
    alpha = Float64(mixture_weight)
    isfinite(alpha) && 0.0 <= alpha < 1.0 || throw(ArgumentError(
        "mixture_weight must be finite and lie in [0, 1).",
    ))
    kappa = Float64(focus_strength)
    isfinite(kappa) && 0.0 <= kappa <= 1.0 || throw(ArgumentError(
        "focus_strength must be finite and lie in [0, 1].",
    ))

    unit_normals = _normalized_geometry_normals(normals, dimension)
    coordinate_second_moment = 1.0 / (dimension + 2.0 * Float64(beta) + 2.0)
    normalizer = 1.0 + kappa^2 * coordinate_second_moment
    focus_ratio = sum(
        (1.0 + kappa * dot(view(unit_normals, :, column), point))^2 / normalizer
        for column in axes(unit_normals, 2)
    ) / size(unit_normals, 2)
    ratio = (1.0 - alpha) + alpha * focus_ratio
    isfinite(ratio) && ratio > 0.0 || error("Internal nonpositive proposal ratio.")
    return ratio
end

function _sample_focused_ball(
    rng::AbstractRNG,
    normals::Matrix{Float64};
    beta::Float64,
    focus_strength::Float64,
)
    dimension = size(normals, 1)
    normal_index = rand(rng, axes(normals, 2))
    normal = view(normals, :, normal_index)
    envelope = (1.0 + focus_strength)^2
    while true
        candidate = sample_ball_smoothing(rng, dimension; beta = beta)
        numerator = (1.0 + focus_strength * dot(normal, candidate))^2
        rand(rng) * envelope <= numerator && return candidate
    end
end

"""
    sample_geometry_smoothing(rng, dimension, count, normals; kwargs...)

Draw exact samples from the geometry-aware mixture proposal and return the
ordinary importance weights `q(u) / p_z(u)`. The estimator must divide by the
declared sample count, not by the sum of weights; these are ordinary importance
weights, not self-normalized importance sampling.
"""
function sample_geometry_smoothing(
    rng::AbstractRNG,
    dimension::Integer,
    count::Integer,
    normals;
    beta::Real = 2.0,
    mixture_weight::Real = 0.5,
    focus_strength::Real = 0.9,
)
    d = Int(dimension)
    n = Int(count)
    d > 0 || throw(ArgumentError("dimension must be positive."))
    n > 0 || throw(ArgumentError("count must be positive."))
    unit_normals = _normalized_geometry_normals(normals, d)
    alpha = Float64(mixture_weight)
    kappa = Float64(focus_strength)
    # Validate all scalar parameters through the public density-ratio path.
    geometry_proposal_ratio(
        zeros(d),
        unit_normals;
        beta = beta,
        mixture_weight = alpha,
        focus_strength = kappa,
    )

    samples = Matrix{Float64}(undef, d, n)
    ratios = Vector{Float64}(undef, n)
    weights = Vector{Float64}(undef, n)
    components = Vector{Symbol}(undef, n)
    for column in 1:n
        focused = rand(rng) < alpha
        sample = focused ? _sample_focused_ball(
            rng,
            unit_normals;
            beta = Float64(beta),
            focus_strength = kappa,
        ) : sample_ball_smoothing(rng, d; beta = beta)
        ratio = geometry_proposal_ratio(
            sample,
            unit_normals;
            beta = beta,
            mixture_weight = alpha,
            focus_strength = kappa,
        )
        samples[:, column] .= sample
        ratios[column] = ratio
        weights[column] = inv(ratio)
        components[column] = focused ? :focused : :base
    end
    return GeometrySampleBatch(
        samples,
        weights,
        ratios,
        components,
        unit_normals,
        alpha,
        kappa,
        Float64(beta),
    )
end

sample_geometry_smoothing(
    dimension::Integer,
    count::Integer,
    normals;
    kwargs...,
) = sample_geometry_smoothing(
    Random.default_rng(), dimension, count, normals; kwargs...,
)

function _representative_competitor_choices(
    assignments::Vector{Int},
    winner::Vector{Int},
)
    return [assignments[index] == 0 ? winner[index] : assignments[index]
            for index in eachindex(assignments)]
end

"""
    switching_geometry(model, center, winner, reference_y, groups; kwargs...)

Construct proposal-equation (21) normals from accepted branch projections at
the center. One deterministic complete representative is used per selected
competitor group. Approximate or incomplete geometry can affect variance but
cannot bias the estimator because `sample_geometry_smoothing` evaluates the
exact density ratio of the proposal it actually uses.
"""
function switching_geometry(
    model::DisjunctiveModel,
    center::AbstractVector{<:Real},
    winner::AbstractVector{<:Integer},
    reference_y::AbstractVector{<:Real},
    groups::AbstractVector{CompetitorGroup};
    solver = nothing,
    feasibility_tol::Real = 1e-6,
    normal_tol::Real = 1e-9,
    max_normals::Int = 8,
)
    max_normals > 0 || throw(ArgumentError("max_normals must be positive."))
    normal_tol >= 0 || throw(ArgumentError("normal_tol must be nonnegative."))
    point = Float64.(collect(center))
    reference = Float64.(collect(reference_y))
    length(point) == model.n_outputs || throw(DimensionMismatch(
        "center dimension does not match the model.",
    ))
    length(reference) == model.n_outputs || throw(DimensionMismatch(
        "reference point dimension does not match the model.",
    ))
    winner_choices = Int.(collect(winner))
    _validate_branch_choices(model, winner_choices)

    ordered_groups = sort(
        collect(groups);
        by = group -> (
            group.distance_margin,
            group.divergence_depth,
            Tuple(group.assignments),
        ),
    )
    normals = Vector{Vector{Float64}}()
    choices_used = Vector{Vector{Int}}()
    seen_choices = Set{Tuple}()
    branch_solves = 0
    accepted = 0
    unresolved = 0
    degenerate = 0

    for group in ordered_groups
        length(normals) >= max_normals && break
        choices = _representative_competitor_choices(group.assignments, winner_choices)
        key = Tuple(choices)
        key in seen_choices && continue
        push!(seen_choices, key)
        branch = project_branch(
            model,
            point,
            choices;
            solver = solver,
            feasibility_tol = feasibility_tol,
        )
        branch_solves += 1
        if !branch.accepted
            unresolved += !branch.proven_infeasible
            continue
        end
        accepted += 1
        direction = branch.y .- reference
        magnitude = norm(direction)
        if !(isfinite(magnitude) && magnitude > normal_tol)
            degenerate += 1
            continue
        end
        push!(normals, direction ./ magnitude)
        push!(choices_used, choices)
    end

    matrix = isempty(normals) ? zeros(model.n_outputs, 0) : reduce(hcat, normals)
    status = !isempty(normals) ? :ok : isempty(groups) ? :no_competitors : :fallback
    return SwitchingGeometry(
        matrix,
        choices_used,
        branch_solves,
        accepted,
        unresolved,
        degenerate,
        status,
    )
end
