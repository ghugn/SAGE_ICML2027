module Phase7Scaling

using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

export BoxUnion
export analytic_blackbox_gradient
export analytic_project
export analytic_sage_gradient
export backend_blackbox_gradient
export backend_sage_unpruned_gradient
export box_projection_jacobian
export branch_count
export dimension
export disjunctive_model
export estimator_summary
export hard_local_gradient
export minimum_center_distance
export project_box
export sample_matrix
export scaling_box_union
export ste_gradient
export task_loss
export task_loss_gradient

"""A deterministic union of separated, axis-aligned boxes."""
struct BoxUnion
    centers::Matrix{Float64} # dimension x branches
    halfwidth::Float64

    function BoxUnion(centers::AbstractMatrix{<:Real}, halfwidth::Real)
        size(centers, 1) > 0 || throw(ArgumentError("dimension must be positive."))
        size(centers, 2) > 1 || throw(ArgumentError("at least two branches are required."))
        isfinite(halfwidth) && halfwidth > 0 ||
            throw(ArgumentError("halfwidth must be finite and positive."))
        all(isfinite, centers) || throw(ArgumentError("centers must be finite."))
        return new(Matrix{Float64}(centers), Float64(halfwidth))
    end
end

dimension(geometry::BoxUnion) = size(geometry.centers, 1)
branch_count(geometry::BoxUnion) = size(geometry.centers, 2)

"""
    scaling_box_union(dimension, branches; extent=1.2, halfwidth=0.25)

Build the first `branches` points of the smallest regular Cartesian grid that
contains them. This gives a deterministic, reproducible scaling family. The
box interiors are disjoint for every Phase 7 smoke-grid configuration.
"""
function scaling_box_union(
    requested_dimension::Integer,
    requested_branches::Integer;
    extent::Real = 1.2,
    halfwidth::Real = 0.25,
)
    d = Int(requested_dimension)
    branches = Int(requested_branches)
    d > 0 || throw(ArgumentError("dimension must be positive."))
    branches > 1 || throw(ArgumentError("branches must exceed one."))
    isfinite(extent) && extent > 0 ||
        throw(ArgumentError("extent must be finite and positive."))

    levels = 2
    while big(levels)^d < branches
        levels += 1
    end
    coordinate_levels = collect(range(-Float64(extent), Float64(extent); length = levels))
    centers = zeros(d, branches)
    for branch in 1:branches
        code = branch - 1
        for coordinate in 1:d
            digit = code % levels
            code = div(code, levels)
            centers[coordinate, branch] = coordinate_levels[digit + 1]
        end
    end
    geometry = BoxUnion(centers, halfwidth)
    minimum_center_distance(geometry, Inf) > 2geometry.halfwidth ||
        error("Generated boxes are not separated.")
    return geometry
end

function minimum_center_distance(geometry::BoxUnion, order::Real = 2)
    minimum_distance = Inf
    for left in 1:(branch_count(geometry) - 1)
        for right in (left + 1):branch_count(geometry)
            minimum_distance = min(
                minimum_distance,
                norm(geometry.centers[:, left] .- geometry.centers[:, right], order),
            )
        end
    end
    return minimum_distance
end

function disjunctive_model(geometry::BoxUnion)
    d = dimension(geometry)
    model = DisjunctiveModel(d)
    bound = maximum(abs, geometry.centers) + geometry.halfwidth + 1.0
    set_bounds!(model, lower = fill(-bound, d), upper = fill(bound, d))
    disjuncts = Vector{LinearConstraint}[]
    for branch in 1:branch_count(geometry)
        constraints = LinearConstraint[]
        for coordinate in 1:d
            normal = zeros(d)
            normal[coordinate] = 1.0
            center = geometry.centers[coordinate, branch]
            push!(constraints, LinearConstraint(normal, :>=, center - geometry.halfwidth))
            push!(constraints, LinearConstraint(normal, :<=, center + geometry.halfwidth))
        end
        push!(disjuncts, constraints)
    end
    add_disjunction!(model, disjuncts...)
    return model
end

function project_box(geometry::BoxUnion, point, branch::Integer)
    center = view(geometry.centers, :, branch)
    return clamp.(point, center .- geometry.halfwidth, center .+ geometry.halfwidth)
end

function analytic_project(geometry::BoxUnion, point; tie_tol::Real = 1e-12)
    length(point) == dimension(geometry) || throw(DimensionMismatch("point dimension mismatch."))
    best_branch = 1
    best_y = project_box(geometry, point, 1)
    best_objective = 0.5 * sum(abs2, best_y .- point)
    for branch in 2:branch_count(geometry)
        candidate_y = project_box(geometry, point, branch)
        candidate_objective = 0.5 * sum(abs2, candidate_y .- point)
        if candidate_objective < best_objective - Float64(tie_tol)
            best_branch = branch
            best_y = candidate_y
            best_objective = candidate_objective
        end
    end
    return (y = collect(best_y), branch = best_branch, objective = best_objective)
end

function box_projection_jacobian(geometry::BoxUnion, point, branch::Integer)
    center = view(geometry.centers, :, branch)
    diagonal = [
        center[index] - geometry.halfwidth < point[index] <
        center[index] + geometry.halfwidth ? 1.0 : 0.0
        for index in eachindex(point)
    ]
    return Diagonal(diagonal)
end

task_loss(y, target) = sum(abs2, y .- target)
task_loss_gradient(y, target) = 2.0 .* (y .- target)

function sample_matrix(
    rng,
    d::Integer,
    count::Integer;
    beta::Real = 2.0,
    antithetic::Bool = false,
)
    count > 0 || throw(ArgumentError("sample count must be positive."))
    antithetic && return sample_antithetic_ball_smoothing(
        rng,
        d,
        count;
        beta = beta,
    )
    return reduce(hcat, [sample_ball_smoothing(rng, d; beta = beta) for _ in 1:count])
end

function hard_local_gradient(geometry::BoxUnion, z, target)
    center = analytic_project(geometry, z)
    return box_projection_jacobian(geometry, z, center.branch)' *
        task_loss_gradient(center.y, target)
end

ste_gradient(geometry::BoxUnion, z, target) =
    task_loss_gradient(analytic_project(geometry, z).y, target)

function analytic_sage_gradient(
    geometry::BoxUnion,
    z,
    radius::Real,
    target,
    samples;
    beta::Real = 2.0,
)
    center = analytic_project(geometry, z)
    total = zeros(length(z))
    switching = 0
    for column in axes(samples, 2)
        u = samples[:, column]
        w = z .+ Float64(radius) .* u
        local_y = project_box(geometry, w, center.branch)
        local_gradient = box_projection_jacobian(geometry, w, center.branch)' *
            task_loss_gradient(local_y, target)
        global_result = analytic_project(geometry, w)
        residual = task_loss(global_result.y, target) - task_loss(local_y, target)
        total .+= local_gradient .+
            (residual / Float64(radius)) .* ball_smoothing_score(u; beta = beta)
        switching += global_result.branch != center.branch
    end
    return (gradient = total ./ size(samples, 2), switching = switching)
end

function analytic_blackbox_gradient(
    geometry::BoxUnion,
    z,
    radius::Real,
    target,
    samples;
    beta::Real = 2.0,
)
    center_loss = task_loss(analytic_project(geometry, z).y, target)
    total = zeros(length(z))
    for column in axes(samples, 2)
        u = samples[:, column]
        global_y = analytic_project(geometry, z .+ Float64(radius) .* u).y
        residual = task_loss(global_y, target) - center_loss
        total .+= (residual / Float64(radius)) .* ball_smoothing_score(u; beta = beta)
    end
    return total ./ size(samples, 2)
end

function _cosine_similarity(left, right)
    denominator = norm(left) * norm(right)
    return denominator == 0.0 ? NaN : dot(left, right) / denominator
end

function estimator_summary(estimates, reference)
    isempty(estimates) && throw(ArgumentError("estimates cannot be empty."))
    norm(reference) > 0 || throw(ArgumentError("reference gradient must be nonzero."))
    mean_estimate = reduce(+, estimates) ./ length(estimates)
    bias = mean_estimate .- reference
    trace_variance = mean(sum(abs2, estimate .- mean_estimate) for estimate in estimates)
    rmse = sqrt(mean(sum(abs2, estimate .- reference) for estimate in estimates))
    cosines = [_cosine_similarity(estimate, reference) for estimate in estimates]
    return (
        mean = mean_estimate,
        relative_bias = norm(bias) / norm(reference),
        trace_variance = trace_variance,
        rmse = rmse,
        mean_cosine = mean(cosines),
        minimum_cosine = minimum(cosines),
    )
end

"""Matched-q black-box backend with the shared center forward excluded from timing."""
function backend_blackbox_gradient(
    model,
    geometry,
    center_result,
    z,
    radius,
    target,
    samples;
    beta::Real = 2.0,
)
    center_loss = task_loss(center_result.y, target)
    total = zeros(length(z))
    leaf_solves = 0
    relaxation_solves = 0
    unsafe = 0
    feasible = true
    started = time_ns()
    for column in axes(samples, 2)
        u = samples[:, column]
        w = z .+ radius .* u
        result = lazy_branch_search(model, w; store_prune_records = false)
        result.status == :optimal || error("Black-box perturbed search was not optimal.")
        exact = analytic_project(geometry, w)
        unsafe += only(result.choices) != exact.branch ||
            !isapprox(result.y, exact.y; atol = 5e-7)
        feasible &= result.report !== nothing && result.report.feasible
        residual = task_loss(result.y, target) - center_loss
        total .+= (residual / radius) .* ball_smoothing_score(u; beta = beta)
        leaf_solves += result.leaf_solves
        relaxation_solves += result.relaxation_solves
    end
    elapsed = (time_ns() - started) / 1e9
    return (
        gradient = total ./ size(samples, 2),
        global_searches = size(samples, 2),
        reference_branch_solves = 0,
        competitor_branch_solves = 0,
        leaf_solves = leaf_solves,
        relaxation_solves = relaxation_solves,
        unsafe_pruning = unsafe,
        feasible = feasible,
        seconds = elapsed,
    )
end

"""
Proposal equations (17)-(18) without certificate or competitor pruning.
Every perturbation performs an unrestricted exact global search.
"""
function backend_sage_unpruned_gradient(
    model,
    geometry,
    center_result,
    z,
    radius,
    target,
    samples;
    beta::Real = 2.0,
)
    total = zeros(length(z))
    switching = 0
    unsafe = 0
    feasible = true
    started = time_ns()
    for column in axes(samples, 2)
        u = samples[:, column]
        w = z .+ radius .* u
        reference = project_branch(model, w, center_result.choices)
        local_result = fixed_branch_loss_gradient(
            model,
            w,
            center_result.choices,
            y -> task_loss_gradient(y, target);
            branch_result = reference,
        )
        local_result.status == :ok || error("Unpruned SAGE local derivative was unresolved.")
        global_result = lazy_branch_search(model, w; store_prune_records = false)
        global_result.status == :optimal || error("Unpruned SAGE global search was unresolved.")
        exact = analytic_project(geometry, w)
        unsafe += only(global_result.choices) != exact.branch ||
            !isapprox(global_result.y, exact.y; atol = 5e-7)
        feasible &= global_result.report !== nothing && global_result.report.feasible
        residual = task_loss(global_result.y, target) - task_loss(local_result.y, target)
        total .+= local_result.gradient .+
            (residual / radius) .* ball_smoothing_score(u; beta = beta)
        switching += global_result.choices != center_result.choices
    end
    elapsed = (time_ns() - started) / 1e9
    return (
        gradient = total ./ size(samples, 2),
        global_searches = size(samples, 2),
        reference_branch_solves = size(samples, 2),
        competitor_branch_solves = 0,
        leaf_solves = 0,
        relaxation_solves = 0,
        unsafe_pruning = unsafe,
        feasible = feasible,
        switching_samples = switching,
        seconds = elapsed,
    )
end

end # module
