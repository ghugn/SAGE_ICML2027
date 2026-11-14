using Random
using LinearAlgebra

"""Per-perturbation diagnostics for proposal equation (18)."""
struct SAGESampleRecord
    u::Vector{Float64}
    w::Vector{Float64}
    local_y::Vector{Float64}
    global_y::Vector{Float64}
    global_choices::Vector{Int}
    local_loss::Float64
    global_loss::Float64
    residual::Float64
    local_gradient::Vector{Float64}
    correction::Vector{Float64}
    gradient::Vector{Float64}
    switched::Bool
    status::Symbol
    importance_weight::Float64
    proposal_ratio::Float64
    sampling_component::Symbol
end

"""Monte Carlo estimate and solver/certificate diagnostics for SAGE."""
struct SAGEEstimate
    gradient::Vector{Float64}
    status::Symbol
    center::Vector{Float64}
    radius::Float64
    winner::Vector{Int}
    certificate::SwitchingCertificate
    samples::Int
    reference_branch_solves::Int
    perturbed_global_searches::Int
    competitor_branch_solves::Int
    sample_stability_certificates::Int
    restricted_global_searches::Int
    lazy_global_searches::Int
    lazy_leaf_solves::Int
    lazy_relaxation_solves::Int
    switching_samples::Int
    unresolved_samples::Int
    sampling::Symbol
    geometry_status::Symbol
    geometry_normals::Int
    geometry_branch_solves::Int
    importance_effective_sample_size::Float64
    use_ball_certificate::Bool
    use_sample_certificate::Bool
    solve_time::Float64
    records::Vector{SAGESampleRecord}
end

"""Typed public configuration for the SAGE research-alpha estimator."""
Base.@kwdef struct SAGEConfig
    radius::Float64
    n_samples::Int = 16
    beta::Float64 = 2.0
    sampling::Symbol = :isotropic
    geometry_mixture_weight::Float64 = 0.5
    geometry_focus_strength::Float64 = 0.9
    geometry_max_normals::Int = 8
    use_ball_certificate::Bool = true
    use_sample_certificate::Bool = true
    global_strategy::Symbol = :auto
    restricted_branch_limit::Int = 4
    store_records::Bool = true
end

is_certified(estimate::SAGEEstimate) = estimate.status == :ok

function _reference_dominates_competitors(
    model::DisjunctiveModel,
    point::Vector{Float64},
    reference::BranchProjectionResult,
    competitor_groups::Vector{CompetitorGroup},
    tolerance::Float64,
)
    # Each geometric value is a certified lower bound for the complete
    # competitor subtree. Strict dominance also preserves deterministic ties.
    for group in competitor_groups
        lower_bound = geometric_lower_bound(model, point, group.assignments)
        lower_bound > reference.objective + tolerance || return false
    end
    return true
end

function _resolve_smoothing_samples(
    rng::AbstractRNG,
    dimension::Int,
    n_samples::Int,
    beta::Float64,
    samples,
)
    if samples === nothing
        n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
        return [sample_ball_smoothing(rng, dimension; beta = beta) for _ in 1:n_samples]
    end
    sample_matrix = Matrix{Float64}(samples)
    size(sample_matrix, 1) == dimension ||
        throw(DimensionMismatch("samples must have one row per model output."))
    size(sample_matrix, 2) > 0 || throw(ArgumentError("samples cannot be empty."))
    vectors = [copy(sample_matrix[:, index]) for index in axes(sample_matrix, 2)]
    for u in vectors
        all(isfinite, u) || throw(ArgumentError("samples must be finite."))
        sum(abs2, u) < 1.0 ||
            throw(ArgumentError("every sample must lie inside the open unit ball."))
    end
    return vectors
end

"""
    sage_gradient(model, z, radius, loss, loss_gradient; kwargs...)

Implement the SAGE residual decomposition from proposal equations (13)-(18).
`loss` is evaluated separately on each feasible projected output;
`loss_gradient(y)` supplies its gradient with respect to that output. When a
stability certificate is inconclusive, `global_strategy=:auto` enumerates only
small certified competitor sets and otherwise runs exact lazy search seeded by
the already-computed fixed-reference projection.
"""
function sage_gradient(
    model::DisjunctiveModel,
    z::AbstractVector{<:Real},
    radius::Real,
    loss,
    loss_gradient;
    n_samples::Int = 16,
    beta::Real = 2.0,
    rng::AbstractRNG = Random.default_rng(),
    samples = nothing,
    center_result = nothing,
    certificate = nothing,
    solver = nothing,
    feasibility_tol::Real = 1e-6,
    active_tol::Real = 1e-6,
    certificate_tol::Real = 1e-9,
    tie_tol::Real = 1e-10,
    max_nodes::Int = 1_000_000,
    max_relevant_branches::Int = 100_000,
    global_strategy::Symbol = :auto,
    restricted_branch_limit::Int = 4,
    store_records::Bool = true,
    sampling::Symbol = :isotropic,
    importance_weights = nothing,
    geometry_normals = nothing,
    geometry_mixture_weight::Real = 0.5,
    geometry_focus_strength::Real = 0.9,
    geometry_max_normals::Int = 8,
    geometry_normal_tol::Real = 1e-9,
    use_ball_certificate::Bool = true,
    use_sample_certificate::Bool = true,
)
    length(z) == model.n_outputs ||
        throw(DimensionMismatch("z dimension does not match the model."))
    all(isfinite, z) || throw(ArgumentError("z must contain only finite values."))
    isfinite(radius) && radius > 0 ||
        throw(ArgumentError("radius must be finite and strictly positive."))
    isfinite(beta) && beta >= 2.0 ||
        throw(ArgumentError("beta must be finite and at least 2."))
    global_strategy in (:auto, :restricted, :lazy) || throw(ArgumentError(
        "global_strategy must be :auto, :restricted, or :lazy.",
    ))
    restricted_branch_limit > 0 ||
        throw(ArgumentError("restricted_branch_limit must be positive."))
    sampling in (:isotropic, :geometry, :provided) || throw(ArgumentError(
        "sampling must be :isotropic, :geometry, or :provided.",
    ))
    sampling == :provided && samples === nothing && throw(ArgumentError(
        "sampling=:provided requires explicit samples.",
    ))
    sampling == :geometry && samples !== nothing && throw(ArgumentError(
        "sampling=:geometry generates its own exact proposal samples; use " *
        "sampling=:provided with explicit importance_weights to reuse a batch.",
    ))
    importance_weights !== nothing && samples === nothing && throw(ArgumentError(
        "importance_weights require explicit samples.",
    ))
    geometry_normals !== nothing && sampling != :geometry && throw(ArgumentError(
        "geometry_normals are valid only with sampling=:geometry.",
    ))
    geometry_max_normals > 0 || throw(ArgumentError(
        "geometry_max_normals must be positive.",
    ))
    geometry_normal_tol >= 0 || throw(ArgumentError(
        "geometry_normal_tol must be nonnegative.",
    ))

    started = time_ns()
    center = Float64.(collect(z))
    center_projection = center_result === nothing ? lazy_branch_search(
        model,
        center;
        solver = solver,
        feasibility_tol = feasibility_tol,
        tie_tol = tie_tol,
        max_nodes = max_nodes,
    ) : center_result
    center_projection isa Union{LazySearchResult, ExactProjectionResult} ||
        throw(ArgumentError("center_result has an unsupported type."))

    computed_stability = switching_certificate(
        model,
        center,
        radius;
        center_result = center_projection,
        solver = solver,
        feasibility_tol = feasibility_tol,
        certificate_tol = certificate_tol,
        tie_tol = tie_tol,
        max_nodes = max_nodes,
    )
    if certificate !== nothing
        certificate isa SwitchingCertificate ||
            throw(ArgumentError("certificate must be a SwitchingCertificate."))
        certificate.center == computed_stability.center &&
        certificate.radius == computed_stability.radius &&
        certificate.status == computed_stability.status &&
        certificate.winner == computed_stability.winner &&
        certificate.stable_pruned_branches == computed_stability.stable_pruned_branches ||
            throw(ArgumentError(
                "certificate is stale or does not match the model, center, and radius.",
            ))
    end
    stability = computed_stability
    stability.center_status == :optimal ||
        return SAGEEstimate(
            Float64[], :center_uncertified, center, Float64(radius), Int[],
            stability, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            sampling, :not_built, 0, 0, 0.0,
            use_ball_certificate, use_sample_certificate,
            (time_ns() - started) / 1e9, SAGESampleRecord[],
        )

    ball_stable = use_ball_certificate && certifies_stability(stability)
    relevant_groups = use_ball_certificate ? competitor_set(stability) : stability.groups
    geometry_status = :not_requested
    geometry_normal_count = 0
    geometry_branch_solves = 0
    sampling_label = sampling
    sample_components = Symbol[]
    sample_weights = Float64[]
    sample_ratios = Float64[]

    smoothing_samples = Vector{Vector{Float64}}()
    if sampling == :geometry && !ball_stable
        normal_matrix = if geometry_normals === nothing
            geometry = switching_geometry(
                model,
                center,
                stability.winner,
                stability.winner_point,
                relevant_groups;
                solver = solver,
                feasibility_tol = feasibility_tol,
                normal_tol = geometry_normal_tol,
                max_normals = geometry_max_normals,
            )
            geometry_status = geometry.status
            geometry_branch_solves = geometry.branch_solves
            geometry.normals
        else
            geometry_status = :provided
            _normalized_geometry_normals(
                geometry_normals,
                model.n_outputs;
                tolerance = Float64(geometry_normal_tol),
            )
        end
        geometry_normal_count = size(normal_matrix, 2)
        if geometry_normal_count > 0
            batch = sample_geometry_smoothing(
                rng,
                model.n_outputs,
                n_samples,
                normal_matrix;
                beta = beta,
                mixture_weight = geometry_mixture_weight,
                focus_strength = geometry_focus_strength,
            )
            smoothing_samples = [copy(batch.samples[:, index])
                                 for index in axes(batch.samples, 2)]
            sample_weights = copy(batch.importance_weights)
            sample_ratios = copy(batch.proposal_ratios)
            sample_components = copy(batch.components)
        else
            # Degenerate or unavailable normals affect only efficiency. Falling
            # back to q preserves the exact objective and estimator target.
            sampling_label = :geometry_fallback
            smoothing_samples = _resolve_smoothing_samples(
                rng, model.n_outputs, n_samples, Float64(beta), nothing,
            )
            sample_weights = ones(length(smoothing_samples))
            sample_ratios = ones(length(smoothing_samples))
            sample_components = fill(:base, length(smoothing_samples))
        end
    else
        if sampling == :geometry
            geometry_status = :stable_fallback
            sampling_label = :geometry_fallback
        end
        smoothing_samples = _resolve_smoothing_samples(
            rng,
            model.n_outputs,
            n_samples,
            Float64(beta),
            samples,
        )
        if importance_weights === nothing
            sample_weights = ones(length(smoothing_samples))
        else
            sample_weights = Float64.(collect(importance_weights))
            length(sample_weights) == length(smoothing_samples) || throw(DimensionMismatch(
                "importance_weights must contain one value per sample.",
            ))
            all(weight -> isfinite(weight) && weight > 0.0, sample_weights) ||
                throw(ArgumentError("importance weights must be finite and positive."))
        end
        sample_ratios = inv.(sample_weights)
        sample_components = fill(samples === nothing ? :base : :provided,
                                 length(smoothing_samples))
    end
    effective_sample_size = sum(sample_weights)^2 / sum(abs2, sample_weights)
    gradient_sum = zeros(model.n_outputs)
    reference_solves = 0
    global_searches = 0
    competitor_solves = 0
    sample_certificates = 0
    restricted_searches = 0
    lazy_searches = 0
    lazy_leaf_solves = 0
    lazy_relaxation_solves = 0
    switching_samples = 0
    unresolved_samples = 0
    records = SAGESampleRecord[]

    for (sample_index, u) in enumerate(smoothing_samples)
        importance_weight = sample_weights[sample_index]
        proposal_ratio = sample_ratios[sample_index]
        sampling_component = sample_components[sample_index]
        w = center .+ Float64(radius) .* u
        reference = project_branch(
            model,
            w,
            stability.winner;
            solver = solver,
            feasibility_tol = feasibility_tol,
        )
        reference_solves += 1
        local_result = fixed_branch_loss_gradient(
            model,
            w,
            stability.winner,
            loss_gradient;
            branch_result = reference,
            solver = solver,
            feasibility_tol = feasibility_tol,
            active_tol = active_tol,
        )
        if local_result.status != :ok
            unresolved_samples += 1
            store_records && push!(records, SAGESampleRecord(
                copy(u), w, Float64[], Float64[], Int[], NaN, NaN, NaN,
                Float64[], Float64[], Float64[], false, :local_unresolved,
                importance_weight, proposal_ratio, sampling_component,
            ))
            continue
        end

        local_loss = Float64(loss(local_result.y))
        isfinite(local_loss) || throw(ArgumentError("loss returned a non-finite value."))
        global_y = local_result.y
        global_choices = stability.winner
        global_loss = local_loss
        residual = 0.0
        switched = false

        if !ball_stable && use_sample_certificate && _reference_dominates_competitors(
            model,
            w,
            reference,
            relevant_groups,
            max(Float64(certificate_tol), Float64(tie_tol)),
        )
            # The fixed reference branch is certified globally optimal at this
            # sampled point. Eq. (18) has zero residual, so no global solve is
            # needed even though the whole perturbation ball was inconclusive.
            sample_certificates += 1
        elseif !ball_stable
            global_searches += 1
            relevant_count = 1 + sum(group.branch_count for group in relevant_groups; init = 0)
            use_restricted = global_strategy == :restricted ||
                (global_strategy == :auto && relevant_count <= restricted_branch_limit)
            global_status = :uncertified
            candidate_y = Float64[]
            candidate_choices = Int[]
            if use_restricted
                relevant = project_relevant_branches(
                    model,
                    w,
                    stability.winner,
                    relevant_groups;
                    reference_result = reference,
                    solver = solver,
                    feasibility_tol = feasibility_tol,
                    tie_tol = tie_tol,
                    max_relevant_branches = max_relevant_branches,
                )
                restricted_searches += 1
                competitor_solves += relevant.solved_branches - 1
                global_status = relevant.status
                candidate_y = relevant.y
                candidate_choices = relevant.choices
            else
                global_result = lazy_branch_search(
                    model,
                    w;
                    solver = solver,
                    feasibility_tol = feasibility_tol,
                    tie_tol = tie_tol,
                    max_nodes = max_nodes,
                    seed_incumbent = false,
                    incumbent_result = reference,
                    store_prune_records = false,
                )
                lazy_searches += 1
                # The externally supplied reference branch is already counted
                # in reference_solves, so only additional leaves are competitors.
                additional_leaves = max(global_result.leaf_solves - 1, 0)
                competitor_solves += additional_leaves
                lazy_leaf_solves += additional_leaves
                lazy_relaxation_solves += global_result.relaxation_solves
                global_status = global_result.status
                candidate_y = global_result.y
                candidate_choices = global_result.choices
            end
            if global_status != :optimal
                unresolved_samples += 1
                store_records && push!(records, SAGESampleRecord(
                    copy(u), w, local_result.y, Float64[], Int[], local_loss, NaN, NaN,
                    local_result.gradient, Float64[], Float64[], false, :global_unresolved,
                    importance_weight, proposal_ratio, sampling_component,
                ))
                continue
            end
            global_y = candidate_y
            global_choices = candidate_choices
            global_loss = Float64(loss(global_y))
            isfinite(global_loss) ||
                throw(ArgumentError("loss returned a non-finite value."))
            residual = global_loss - local_loss
            switched = global_choices != stability.winner
            switched && (switching_samples += 1)
        end

        score = ball_smoothing_score(u; beta = beta)
        correction = (residual / Float64(radius)) .* score
        sample_gradient = importance_weight .* (local_result.gradient .+ correction)
        gradient_sum .+= sample_gradient
        store_records && push!(records, SAGESampleRecord(
            copy(u), w, local_result.y, copy(global_y), copy(global_choices),
            local_loss, global_loss, residual, copy(local_result.gradient), correction,
            sample_gradient, switched, :ok,
            importance_weight, proposal_ratio, sampling_component,
        ))
    end

    status = unresolved_samples == 0 ? :ok : :uncertified
    gradient = status == :ok ? gradient_sum ./ length(smoothing_samples) : Float64[]
    return SAGEEstimate(
        gradient,
        status,
        center,
        Float64(radius),
        copy(stability.winner),
        stability,
        length(smoothing_samples),
        reference_solves,
        global_searches,
        competitor_solves,
        sample_certificates,
        restricted_searches,
        lazy_searches,
        lazy_leaf_solves,
        lazy_relaxation_solves,
        switching_samples,
        unresolved_samples,
        sampling_label,
        geometry_status,
        geometry_normal_count,
        geometry_branch_solves,
        effective_sample_size,
        use_ball_certificate,
        use_sample_certificate,
        (time_ns() - started) / 1e9,
        records,
    )
end

"""Configuration-object overload for the research-alpha public API."""
function sage_gradient(
    model::DisjunctiveModel,
    z::AbstractVector{<:Real},
    loss,
    loss_gradient,
    config::SAGEConfig;
    kwargs...,
)
    return sage_gradient(
        model,
        z,
        config.radius,
        loss,
        loss_gradient;
        n_samples = config.n_samples,
        beta = config.beta,
        sampling = config.sampling,
        geometry_mixture_weight = config.geometry_mixture_weight,
        geometry_focus_strength = config.geometry_focus_strength,
        geometry_max_normals = config.geometry_max_normals,
        use_ball_certificate = config.use_ball_certificate,
        use_sample_certificate = config.use_sample_certificate,
        global_strategy = config.global_strategy,
        restricted_branch_limit = config.restricted_branch_limit,
        store_records = config.store_records,
        kwargs...,
    )
end

