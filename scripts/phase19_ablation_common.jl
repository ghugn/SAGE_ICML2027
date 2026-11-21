module Phase19Ablation

using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

if !isdefined(parentmodule(@__MODULE__), :Phase10JobShop)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "phase10_jobshop_common.jl"))
end
using ..Phase10JobShop
const JS = Phase10JobShop

export PHASE19_VARIANTS
export ablate_task
export center_context
export reference_gradient
export run_variant

const PHASE19_VARIANTS = (
    :compact_blackbox,
    :residual_decomposition,
    :ball_certificate,
    :safe_pruning,
    :boundary_geometry,
    :certified_multiscale,
)

"""Fixed exact-forward context shared by every estimator variant."""
function center_context(problem, network, index::Integer)
    1 <= index <= size(problem.train.inputs, 2) || throw(BoundsError(
        axes(problem.train.inputs, 2), index,
    ))
    point, _ = JS._forward(network, problem.train.inputs[:, index])
    instance = problem.train.instances[index]
    center = lazy_branch_search(
        instance.model, point; store_prune_records = false,
    )
    center.status == :optimal || error("Phase 19 center projection is unresolved.")
    center.report !== nothing && center.report.feasible ||
        error("Phase 19 center projection is logically infeasible.")
    return (
        problem = problem,
        network = network,
        index = Int(index),
        point = point,
        instance = instance,
        center = center,
        center_loss = JS._task_loss(center.y, instance, problem.config),
    )
end

function _parallel_columns(evaluate, count::Int, parallel::Bool)
    results = Vector{Any}(undef, count)
    if parallel && Threads.nthreads() > 1
        Threads.@threads :static for column in 1:count
            results[column] = evaluate(column)
        end
    else
        for column in 1:count
            results[column] = evaluate(column)
        end
    end
    return results
end

function _compact_blackbox(context, samples, radius; parallel::Bool)
    started = time_ns()
    count = size(samples, 2)
    results = _parallel_columns(count, parallel) do column
        direction = samples[:, column]
        projected = lazy_branch_search(
            context.instance.model,
            context.point .+ radius .* direction;
            store_prune_records = false,
        )
        projected.status == :optimal || error("Compact black-box search unresolved.")
        feasible = projected.report !== nothing && projected.report.feasible
        residual = JS._task_loss(projected.y, context.instance, context.problem.config) -
            context.center_loss
        term = (residual / radius) .* ball_smoothing_score(direction; beta = 2.0)
        return (
            gradient = term,
            global_searches = 1,
            branch_solves = projected.leaf_solves,
            relaxation_solves = projected.relaxation_solves,
            switched = projected.choices != context.center.choices,
            unresolved = 0,
            logical_failures = feasible ? 0 : 1,
            worker = Threads.threadid(),
        )
    end
    return (
        gradient = reduce(+, (row.gradient for row in results)) ./ count,
        seconds = (time_ns() - started) / 1e9,
        global_searches = sum(row.global_searches for row in results),
        branch_solves = sum(row.branch_solves for row in results),
        relaxation_solves = sum(row.relaxation_solves for row in results),
        switching_samples = sum(row.switched for row in results),
        unresolved = sum(row.unresolved for row in results),
        logical_failures = sum(row.logical_failures for row in results),
        sample_certificates = 0,
        ball_certified = false,
        pruned_fraction = 0.0,
        ess = Float64(count),
        workers = length(unique(row.worker for row in results)),
    )
end

"""Proposal Eq. (17)-(18) without ball/sample certificates or competitor pruning."""
function _residual_decomposition(context, samples, radius; parallel::Bool)
    started = time_ns()
    count = size(samples, 2)
    results = _parallel_columns(count, parallel) do column
        direction = samples[:, column]
        perturbed = context.point .+ radius .* direction
        reference = project_branch(
            context.instance.model, perturbed, context.center.choices,
        )
        local_result = fixed_branch_loss_gradient(
            context.instance.model,
            perturbed,
            context.center.choices,
            y -> JS._task_loss_gradient(y, context.instance, context.problem.config);
            branch_result = reference,
        )
        local_result.status == :ok || error("Residual-decomposition local solve unresolved.")
        global_result = lazy_branch_search(
            context.instance.model,
            perturbed;
            seed_incumbent = false,
            incumbent_result = reference,
            store_prune_records = false,
        )
        global_result.status == :optimal || error("Residual-decomposition global solve unresolved.")
        feasible = global_result.report !== nothing && global_result.report.feasible
        local_loss = JS._task_loss(local_result.y, context.instance, context.problem.config)
        global_loss = JS._task_loss(global_result.y, context.instance, context.problem.config)
        correction = ((global_loss - local_loss) / radius) .*
            ball_smoothing_score(direction; beta = 2.0)
        # The lazy result includes the supplied reference incumbent. Count the
        # fixed-reference QP once and only the additional exact branch leaves.
        additional = max(global_result.leaf_solves - 1, 0)
        return (
            gradient = local_result.gradient .+ correction,
            global_searches = 1,
            branch_solves = 1 + additional,
            relaxation_solves = global_result.relaxation_solves,
            switched = global_result.choices != context.center.choices,
            unresolved = 0,
            logical_failures = feasible ? 0 : 1,
            worker = Threads.threadid(),
        )
    end
    return (
        gradient = reduce(+, (row.gradient for row in results)) ./ count,
        seconds = (time_ns() - started) / 1e9,
        global_searches = sum(row.global_searches for row in results),
        branch_solves = sum(row.branch_solves for row in results),
        relaxation_solves = sum(row.relaxation_solves for row in results),
        switching_samples = sum(row.switched for row in results),
        unresolved = sum(row.unresolved for row in results),
        logical_failures = sum(row.logical_failures for row in results),
        sample_certificates = 0,
        ball_certified = false,
        pruned_fraction = 0.0,
        ess = Float64(count),
        workers = length(unique(row.worker for row in results)),
    )
end

function _certificate_fraction(certificate)
    total = certificate.total_competitor_branches
    return total == 0 ? 1.0 : certificate.stable_pruned_branches / total
end

function _from_sage_estimate(estimate, seconds; extra_geometry_solves::Int = 0)
    return (
        gradient = estimate.gradient,
        seconds = seconds,
        global_searches = estimate.perturbed_global_searches,
        # `competitor_branch_solves` already contains additional lazy leaves.
        # Do not add `lazy_leaf_solves` a second time.
        branch_solves = estimate.reference_branch_solves +
            estimate.competitor_branch_solves + estimate.geometry_branch_solves +
            extra_geometry_solves,
        relaxation_solves = estimate.lazy_relaxation_solves,
        switching_samples = estimate.switching_samples,
        unresolved = estimate.unresolved_samples,
        logical_failures = 0,
        sample_certificates = estimate.sample_stability_certificates,
        ball_certified = certifies_stability(estimate.certificate),
        pruned_fraction = _certificate_fraction(estimate.certificate),
        ess = estimate.importance_effective_sample_size,
        workers = estimate.sample_threads_used,
    )
end

function _sage_variant(variant, context, samples, radius, rng; parallel::Bool)
    loss = y -> JS._task_loss(y, context.instance, context.problem.config)
    loss_gradient = y -> JS._task_loss_gradient(y, context.instance, context.problem.config)
    started = time_ns()
    estimate = if variant == :ball_certificate
        sage_gradient(
            context.instance.model, context.point, radius, loss, loss_gradient;
            samples = samples,
            center_result = context.center,
            use_ball_certificate = true,
            use_sample_certificate = false,
            global_strategy = :lazy,
            store_records = false,
            parallel_samples = parallel,
        )
    elseif variant == :safe_pruning
        sage_gradient(
            context.instance.model, context.point, radius, loss, loss_gradient;
            samples = samples,
            center_result = context.center,
            use_ball_certificate = true,
            use_sample_certificate = true,
            global_strategy = :auto,
            store_records = false,
            parallel_samples = parallel,
        )
    elseif variant == :boundary_geometry
        sage_gradient(
            context.instance.model, context.point, loss, loss_gradient,
            SAGEConfig(
                radius = radius,
                n_samples = size(samples, 2),
                beta = 2.0,
                sampling = :geometry,
                store_records = false,
                parallel_samples = parallel,
            );
            rng = rng,
            center_result = context.center,
        )
    else
        throw(ArgumentError("Unsupported direct SAGE variant $(variant)."))
    end
    estimate.status == :ok || error("$(variant) estimator unresolved: $(estimate.status).")
    return _from_sage_estimate(estimate, (time_ns() - started) / 1e9)
end

function _multiscale(context, samples, radius, rng; parallel::Bool)
    counters = JS.TrainingCounters()
    started = time_ns()
    estimate = JS._multiscale_sage_estimate(
        context.instance,
        context.point,
        samples,
        radius,
        context.problem.config,
        counters,
        rng,
        context.center;
        parallel_samples = parallel,
    )
    estimate.status == :ok || error("Certified-multiscale estimator unresolved.")
    return _from_sage_estimate(
        estimate,
        (time_ns() - started) / 1e9;
        extra_geometry_solves = counters.geometry_branch_solves,
    )
end

"""Evaluate exactly one registered ablation variant at a fixed center."""
function run_variant(
    variant::Symbol,
    context,
    samples::AbstractMatrix,
    radius::Real;
    rng::AbstractRNG = Random.default_rng(),
    parallel::Bool = true,
)
    variant in PHASE19_VARIANTS || throw(ArgumentError("Unknown Phase 19 variant $(variant)."))
    size(samples, 1) == length(context.point) || throw(DimensionMismatch(
        "sample dimension does not match network output dimension.",
    ))
    size(samples, 2) > 0 || throw(ArgumentError("at least one sample is required."))
    result = if variant == :compact_blackbox
        _compact_blackbox(context, samples, Float64(radius); parallel = parallel)
    elseif variant == :residual_decomposition
        _residual_decomposition(context, samples, Float64(radius); parallel = parallel)
    elseif variant in (:ball_certificate, :safe_pruning, :boundary_geometry)
        _sage_variant(
            variant, context, samples, Float64(radius), rng; parallel = parallel,
        )
    else
        _multiscale(context, samples, Float64(radius), rng; parallel = parallel)
    end
    all(isfinite, result.gradient) || error("$(variant) produced a non-finite gradient.")
    result.unresolved == 0 || error("$(variant) left unresolved perturbations.")
    result.logical_failures == 0 || error("$(variant) produced an infeasible output.")
    return result
end

"""High-sample compact-q reference split into independent halves."""
function reference_gradient(
    context,
    seed::Integer;
    radius::Real,
    samples::Int,
    parallel::Bool = true,
)
    samples > 0 && samples % 4 == 0 || throw(ArgumentError(
        "reference samples must be positive and divisible by four.",
    ))
    half = samples ÷ 2
    left_samples = sample_antithetic_ball_smoothing(
        MersenneTwister(Int(seed)), length(context.point), half; beta = 2.0,
    )
    right_samples = sample_antithetic_ball_smoothing(
        MersenneTwister(Int(seed) + 1), length(context.point), half; beta = 2.0,
    )
    left = run_variant(
        :compact_blackbox, context, left_samples, radius;
        rng = MersenneTwister(Int(seed) + 2), parallel = parallel,
    )
    right = run_variant(
        :compact_blackbox, context, right_samples, radius;
        rng = MersenneTwister(Int(seed) + 3), parallel = parallel,
    )
    gradient = 0.5 .* (left.gradient .+ right.gradient)
    scale = max(norm(gradient), eps(Float64))
    return (
        gradient = gradient,
        half_disagreement = norm(left.gradient - right.gradient) / scale,
        seconds = left.seconds + right.seconds,
        global_searches = left.global_searches + right.global_searches,
        branch_solves = left.branch_solves + right.branch_solves,
        workers = max(left.workers, right.workers),
    )
end

function _cosine(left, right)
    denominator = norm(left) * norm(right)
    return denominator <= eps(Float64) ? NaN : dot(left, right) / denominator
end

"""Mean of finite values, or `NaN` when a metric is undefined for every row."""
function _finite_mean(values)
    total = 0.0
    count = 0
    for value in values
        if isfinite(value)
            total += value
            count += 1
        end
    end
    return count == 0 ? NaN : total / count
end

"""
Run the complete registered ablation at one center. Isotropic variants share
the exact same compact-q draws in every repetition; geometry variants use a
separate deterministic stream because their proposal distribution differs.
"""
function ablate_task(
    context,
    seed::Integer;
    radius::Real = 1.0,
    estimator_samples::Int = 4,
    repetitions::Int = 16,
    reference_samples::Int = 256,
    parallel::Bool = true,
    progress = nothing,
)
    estimator_samples > 0 && iseven(estimator_samples) || throw(ArgumentError(
        "estimator_samples must be a positive even integer.",
    ))
    repetitions > 1 || throw(ArgumentError("repetitions must exceed one."))
    reference = reference_gradient(
        context,
        19_000_000 + Int(seed);
        radius = radius,
        samples = reference_samples,
        parallel = parallel,
    )
    raw = NamedTuple[]
    for repetition in 1:repetitions
        stream = 19_100_000 + 10_000Int(seed) + repetition
        samples = sample_antithetic_ball_smoothing(
            MersenneTwister(stream),
            length(context.point),
            estimator_samples;
            beta = 2.0,
        )
        for (variant_index, variant) in enumerate(PHASE19_VARIANTS)
            estimate = run_variant(
                variant,
                context,
                samples,
                radius;
                rng = MersenneTwister(stream + 100variant_index),
                parallel = parallel,
            )
            reference_norm = max(norm(reference.gradient), eps(Float64))
            push!(raw, merge(estimate, (
                variant = variant,
                repetition = repetition,
                relative_error = norm(estimate.gradient - reference.gradient) / reference_norm,
                cosine = _cosine(estimate.gradient, reference.gradient),
                gradient_norm = norm(estimate.gradient),
            )))
            progress === nothing || progress(variant, repetition, repetitions, estimate)
        end
    end

    summaries = NamedTuple[]
    for variant in PHASE19_VARIANTS
        rows = filter(row -> row.variant == variant, raw)
        gradients = [row.gradient for row in rows]
        mean_gradient = reduce(+, gradients) ./ length(gradients)
        deviations = [gradient - mean_gradient for gradient in gradients]
        reference_norm = max(norm(reference.gradient), eps(Float64))
        push!(summaries, (
            variant = variant,
            relative_bias = norm(mean_gradient - reference.gradient) / reference_norm,
            trace_variance = sum(sum(abs2, deviation) for deviation in deviations) /
                (length(deviations) - 1),
            rmse = sqrt(mean(sum(abs2, gradient - reference.gradient) for gradient in gradients)),
            # A zero reference or estimate makes cosine undefined. This is a
            # valid degenerate center, so retain its other metrics and report
            # NaN rather than reducing an empty filtered generator.
            mean_cosine = _finite_mean(row.cosine for row in rows),
            mean_seconds = mean(row.seconds for row in rows),
            mean_global_searches = mean(row.global_searches for row in rows),
            mean_branch_solves = mean(row.branch_solves for row in rows),
            mean_sample_certificates = mean(row.sample_certificates for row in rows),
            ball_certificate_rate = mean(row.ball_certified for row in rows),
            mean_pruned_fraction = mean(row.pruned_fraction for row in rows),
            mean_ess = mean(row.ess for row in rows),
            maximum_workers = maximum(row.workers for row in rows),
        ))
    end
    return (reference = reference, raw = raw, summaries = summaries)
end

end
