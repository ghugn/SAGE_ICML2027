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
    solve_time::Float64
    records::Vector{SAGESampleRecord}
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
            (time_ns() - started) / 1e9, SAGESampleRecord[],
        )

    smoothing_samples = _resolve_smoothing_samples(
        rng,
        model.n_outputs,
        n_samples,
        Float64(beta),
        samples,
    )
    relevant_groups = competitor_set(stability)
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

    for u in smoothing_samples
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

        if !certifies_stability(stability) && _reference_dominates_competitors(
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
        elseif !certifies_stability(stability)
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
        sample_gradient = local_result.gradient .+ correction
        gradient_sum .+= sample_gradient
        store_records && push!(records, SAGESampleRecord(
            copy(u), w, local_result.y, copy(global_y), copy(global_choices),
            local_loss, global_loss, residual, copy(local_result.gradient), correction,
            sample_gradient, switched, :ok,
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
        (time_ns() - started) / 1e9,
        records,
    )
end

