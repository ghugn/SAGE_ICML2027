module Phase18External

using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

if !isdefined(parentmodule(@__MODULE__), :Phase10JobShop)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "phase10_jobshop_common.jl"))
end
using ..Phase10JobShop
const JS = Phase10JobShop

export dbb_projection_vjp, gaussian_perturbed_vjp, external_backend
export setup_problem, train_external, candidate_specs, default_spec, network_equal

"""
Loss-augmented DBB adaptation to Euclidean projection on a union of polyhedra.
P(z-lambda*v) minimizes 0.5||y-z||^2 + lambda*dot(v,y).
The input-z surrogate VJP is (P(z)-P(z-lambda*v))/lambda.
This is not a claim of reproducing the finite-linear-output DBB theorem.
"""
function dbb_projection_vjp(project, point, upstream; lambda::Real, center = nothing)
    isfinite(lambda) && lambda > 0 || throw(ArgumentError("lambda must be finite and positive."))
    length(point) == length(upstream) || throw(DimensionMismatch("point/upstream dimensions differ."))
    all(isfinite, point) && all(isfinite, upstream) || throw(ArgumentError("non-finite input."))
    base = center === nothing ? project(point) : center
    return (base .- project(point .- lambda .* upstream)) ./ lambda
end

"""
Gaussian score VJP of z -> E[P(z+sigma*epsilon)] with frozen upstream v.
Each column of directions is one epsilon; an antithetic pair counts as two
solver calls. The center subtraction is a zero-mean control variate.
It differs from smoothing the composed task loss E[L(P(z+noise))].
"""
function gaussian_perturbed_vjp(project, point, upstream, directions;
        sigma::Real, center = nothing, parallel::Bool = false)
    isfinite(sigma) && sigma > 0 || throw(ArgumentError("sigma must be finite and positive."))
    size(directions, 1) == length(point) == length(upstream) ||
        throw(DimensionMismatch("Gaussian VJP dimensions differ."))
    size(directions, 2) > 0 || throw(ArgumentError("at least one direction is required."))
    all(isfinite, point) && all(isfinite, upstream) && all(isfinite, directions) ||
        throw(ArgumentError("non-finite Gaussian VJP input."))
    base = center === nothing ? project(point) : center
    count = size(directions, 2)
    terms = Vector{Vector{Float64}}(undef, count)
    workers = zeros(Int, count)
    function evaluate(column)
        epsilon = directions[:, column]
        displacement = project(point .+ sigma .* epsilon) .- base
        terms[column] = epsilon .* (dot(displacement, upstream) / sigma)
        workers[column] = Threads.threadid()
    end
    if parallel && Threads.nthreads() > 1
        Threads.@threads :static for column in 1:count
            evaluate(column)
        end
    else
        for column in 1:count
            evaluate(column)
        end
    end
    # Fixed-order reduction makes serial/threaded outputs identical.
    return (gradient = reduce(+, terms) ./ count, workers = length(unique(workers)))
end

network_equal(left, right) = all(
    getproperty(left, name) == getproperty(right, name)
    for name in (:W1, :b1, :W2, :b2, :W3, :b3)
)

"""Separate validation and held-out evaluation instance pools; train-only normalization."""
function setup_problem(seed; pool::Symbol = :validation, small::Bool = false)
    pool in (:validation, :heldout) || throw(ArgumentError("unknown evaluation pool."))
    config = JobShopConfig(n_jobs = 3, n_machines = small ? 3 : 4)
    train_size, evaluation_size = small ? (4, 4) : (16, pool == :validation ? 12 : 32)
    train_raw = JS._raw_split(config, train_size, 1_010_000 + seed, :train)
    iid_seed, ood_seed = pool == :validation ?
        (18_020_002, 18_020_002) : (18_040_004, 18_050_005)
    iid_raw = JS._raw_split(config, evaluation_size, iid_seed, :iid)
    # Validation mode never constructs an OOD test pool.
    ood_raw = pool == :validation ? iid_raw :
        JS._raw_split(config, evaluation_size, ood_seed, :ood)
    center = vec(mean(train_raw.raw_inputs; dims = 2))
    scale = max.(vec(std(train_raw.raw_inputs; dims = 2, corrected = true)), 0.1)
    problem = JobShopProblem(config,
        JS._normalized_split(train_raw, center, scale),
        JS._normalized_split(iid_raw, center, scale),
        JS._normalized_split(ood_raw, center, scale), center, scale)
    pretrained = pretrain_network(problem, seed; epochs = small ? 1 : 8,
        batch_size = min(8, train_size), learning_rate = 2e-3)
    return problem, pretrained
end

function default_spec(label)
    label in (:hard_local, :ste, :compact_blackbox, :sage_n4,
        :dbb_projection, :dpo_gaussian) || throw(ArgumentError("unknown method $(label)."))
    return (label = label, learning_rate = 1e-3, radius = 0.5, samples = 4,
        lambda = 0.5, gaussian_scale = 1.0)
end

# Equal count and update budget, with method-specific search spaces declared
# before test evaluation. rho is fixed throughout each SAGE run, not adapted
# using test outcomes. Gaussian scale is relative to the RMS-matched default.
function candidate_specs(label)
    base = default_spec(label)
    learning_rates = (3e-4, 1e-3, 3e-3)
    if label == :dbb_projection
        return [merge(base, (learning_rate = lr, lambda = lambda))
            for lr in learning_rates for lambda in (0.05, 0.5, 5.0)]
    elseif label == :dpo_gaussian
        return [merge(base, (learning_rate = lr, gaussian_scale = scale))
            for lr in learning_rates for scale in (0.5, 1.0, 2.0)]
    elseif label in (:sage_n4, :compact_blackbox)
        return [merge(base, (learning_rate = lr, radius = radius))
            for lr in learning_rates for radius in (0.25, 0.5, 1.0)]
    end
    return [merge(base, (learning_rate = lr,)) for lr in
        (3e-4, 4e-4, 5.5e-4, 7.5e-4, 1e-3, 1.35e-3, 1.8e-3, 2.4e-3, 3e-3)]
end

function external_backend(spec)
    return function(method, instance, point, perturbations, radius, config, counters, rng;
            adaptive_policy = nothing, parallel_sage_samples = false)
        if spec.label in (:hard_local, :ste, :sage_n4)
            backend = spec.label == :sage_n4 ? :sage_multiscale : spec.label
            return JS._gradient_for_sample(backend, instance, point, perturbations,
                radius, config, counters, rng; parallel_sage_samples = parallel_sage_samples)
        end
        center = lazy_branch_search(instance.model, point; store_prune_records = false)
        center.status == :optimal || error("External baseline center unresolved.")
        JS._record_center!(counters, center)
        loss = JS._task_loss(center.y, instance, config)
        upstream = JS._task_loss_gradient(center.y, instance, config)
        results = Vector{Any}(undef, spec.label == :dbb_projection ? 1 : spec.samples)
        # Each callback receives its own slot; Gaussian threads write disjoint
        # result slots, and all counters are reduced afterwards on the caller.
        function project_at(point_value, index)
            result = lazy_branch_search(instance.model, point_value; store_prune_records = false)
            result.status == :optimal || error("External baseline perturbed search unresolved.")
            result.report !== nothing && result.report.feasible ||
                error("External baseline perturbed output is infeasible.")
            results[index] = result
            return result.y
        end
        workers = 1
        if spec.label == :dbb_projection
            gradient = dbb_projection_vjp(z -> project_at(z, 1), point, upstream;
                lambda = spec.lambda, center = center.y)
        else
            directions = if spec.label == :dpo_gaussian
                half = randn(rng, length(point), spec.samples ÷ 2)
                hcat(half, -half)
            else
                perturbations
            end
            # Compact q on the d-ball with beta=2 has E||u||^2=d/(d+6).
            # Thus Gaussian sigma=rho/sqrt(d+6) matches its RMS displacement.
            sigma = spec.gaussian_scale * spec.radius / sqrt(length(point) + 6)
            terms = Vector{Vector{Float64}}(undef, spec.samples)
            worker_ids = zeros(Int, spec.samples)
            function sample_at(column)
                direction = directions[:, column]
                scale = spec.label == :dpo_gaussian ? sigma : spec.radius
                output = project_at(point .+ scale .* direction, column)
                terms[column] = spec.label == :dpo_gaussian ?
                    direction .* (dot(output .- center.y, upstream) / sigma) :
                    ball_smoothing_score(direction; beta = 2.0) .*
                        ((JS._task_loss(output, instance, config) - loss) / scale)
                worker_ids[column] = Threads.threadid()
            end
            if parallel_sage_samples && Threads.nthreads() > 1
                Threads.@threads :static for column in 1:spec.samples
                    sample_at(column)
                end
            else
                for column in 1:spec.samples
                    sample_at(column)
                end
            end
            gradient = reduce(+, terms) ./ spec.samples
            workers = length(unique(worker_ids))
        end
        for result in results
            counters.perturbed_global_searches += 1
            counters.lazy_leaf_solves += result.leaf_solves
            counters.lazy_relaxation_solves += result.relaxation_solves
            counters.switching_samples += result.choices != center.choices
        end
        counters.sample_threads_used = max(counters.sample_threads_used, workers)
        return gradient, loss
    end
end

function train_external(problem, pretrained, spec, seed; steps, parallel = true, kwargs...)
    default_spec(spec.label) # reject unsupported method labels
    all(value -> isfinite(value) && value > 0,
        (spec.learning_rate, spec.radius, spec.lambda, spec.gaussian_scale)) ||
        throw(ArgumentError("Learning rate and estimator scales must be finite and positive."))
    spec.samples > 0 && iseven(spec.samples) ||
        throw(ArgumentError("samples must be a positive even integer."))
    # Use the original sampler path for both compact smoothing controls.
    method = spec.label == :sage_n4 ? :sage_multiscale :
        spec.label == :compact_blackbox ? :blackbox : spec.label
    return train_method(problem, pretrained, method, seed;
        epochs = cld(steps, size(problem.train.inputs, 2)), max_steps = steps,
        samples = spec.samples, radius = spec.radius, learning_rate = spec.learning_rate,
        parallel_sage_samples = parallel, gradient_backend = external_backend(spec),
        evaluation_splits = :none, evaluation_interval = 64, kwargs...)
end

end
