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
    switching_samples::Int
    unresolved_samples::Int
    solve_time::Float64
    records::Vector{SAGESampleRecord}
end

is_certified(estimate::SAGEEstimate) = estimate.status == :ok

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
`loss_gradient(y)` supplies its gradient with respect to that output.
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
    store_records::Bool = true,
)
    length(z) == model.n_outputs ||
        throw(DimensionMismatch("z dimension does not match the model."))
    all(isfinite, z) || throw(ArgumentError("z must contain only finite values."))
    isfinite(radius) && radius > 0 ||
        throw(ArgumentError("radius must be finite and strictly positive."))
    isfinite(beta) && beta >= 2.0 ||
        throw(ArgumentError("beta must be finite and at least 2."))

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
            stability, 0, 0, 0, 0, 0, 0,
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

        if !certifies_stability(stability)
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
            global_searches += 1
            competitor_solves += relevant.solved_branches - 1
            if relevant.status != :optimal
                unresolved_samples += 1
                store_records && push!(records, SAGESampleRecord(
                    copy(u), w, local_result.y, Float64[], Int[], local_loss, NaN, NaN,
                    local_result.gradient, Float64[], Float64[], false, :global_unresolved,
                ))
                continue
            end
            global_y = relevant.y
            global_choices = relevant.choices
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
        switching_samples,
        unresolved_samples,
        (time_ns() - started) / 1e9,
        records,
    )
end

