module Phase10JobShop

using DisjunctiveNet
using HiGHS
using JuMP
using LinearAlgebra
import MathOptInterface as MOI
using Random
using Statistics

export JobShopConfig
export JobShopInstance
export JobShopMLP
export JobShopProblem
export JobShopSplit
export build_jobshop_model
export evaluate_network
export initialize_network
export prepare_problem
export pretrain_network
export train_method

Base.@kwdef struct JobShopConfig
    n_jobs::Int = 3
    n_machines::Int = 3
    train_duration_min::Int = 2
    train_duration_max::Int = 8
    ood_duration_min::Int = 1
    ood_duration_max::Int = 12
    hidden::Int = 32
    makespan_loss_weight::Float64 = 0.25
    feasibility_tolerance::Float64 = 1e-6
end

struct JobShopInstance
    durations::Vector{Float64}
    raw_durations::Vector{Int}
    machines::Vector{Int}
    horizon::Int
    model::DisjunctiveModel
    oracle_starts::Vector{Float64}
    oracle_choices::Vector{Int}
    oracle_makespan::Float64
    heuristic_starts::Vector{Float64}
    heuristic_makespan::Float64
end

struct JobShopSplit
    raw_inputs::Matrix{Float64}
    inputs::Matrix{Float64}
    targets::Matrix{Float64}
    instances::Vector{JobShopInstance}
end

struct JobShopProblem
    config::JobShopConfig
    train::JobShopSplit
    iid::JobShopSplit
    ood::JobShopSplit
    input_mean::Vector{Float64}
    input_scale::Vector{Float64}
end

mutable struct JobShopMLP
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}
end

mutable struct AdamState
    first::Vector{Array{Float64}}
    second::Vector{Array{Float64}}
    step::Int
end

mutable struct TrainingCounters
    center_global_searches::Int
    center_leaf_solves::Int
    center_relaxation_solves::Int
    perturbed_global_searches::Int
    reference_branch_solves::Int
    competitor_branch_solves::Int
    geometry_branch_solves::Int
    sample_certificates::Int
    lazy_leaf_solves::Int
    lazy_relaxation_solves::Int
    switching_samples::Int
    unresolved_samples::Int
    logical_failures::Int
    effective_sample_size_sum::Float64
    effective_sample_size_count::Int
    base_proposal_samples::Int
    local_proposal_samples::Int
    focused_proposal_samples::Int
    local_radius_sum::Float64
    local_radius_count::Int
    positive_local_radius_calls::Int
end

TrainingCounters() = TrainingCounters(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0,
    0, 0, 0, 0.0, 0, 0,
)

_operation(config::JobShopConfig, job, position) =
    (job - 1) * config.n_machines + position

function _validate_config(config::JobShopConfig)
    config.n_jobs >= 2 || throw(ArgumentError("Job-shop requires at least two jobs."))
    config.n_machines >= 2 || throw(ArgumentError("Job-shop requires at least two machines."))
    config.n_jobs == 3 && config.n_machines == 3 || throw(ArgumentError(
        "Phase 10 controlled implementation is frozen to 3 jobs and 3 machines.",
    ))
    1 <= config.train_duration_min <= config.train_duration_max ||
        throw(ArgumentError("Invalid train duration range."))
    1 <= config.ood_duration_min <= config.ood_duration_max ||
        throw(ArgumentError("Invalid OOD duration range."))
    config.hidden > 0 || throw(ArgumentError("hidden must be positive."))
    config.makespan_loss_weight >= 0 ||
        throw(ArgumentError("makespan_loss_weight must be nonnegative."))
    return nothing
end

function build_jobshop_model(
    normalized_durations,
    machines,
    config::JobShopConfig = JobShopConfig(),
)
    _validate_config(config)
    n_operations = config.n_jobs * config.n_machines
    length(normalized_durations) == n_operations ||
        throw(DimensionMismatch("Unexpected duration vector length."))
    length(machines) == n_operations ||
        throw(DimensionMismatch("Unexpected machine-route vector length."))
    all(machine -> 1 <= machine <= config.n_machines, machines) ||
        throw(ArgumentError("Machine indices are out of range."))

    model = DisjunctiveModel(n_operations)
    set_bounds!(model; lower = zeros(n_operations), upper = ones(n_operations))

    for operation in 1:n_operations
        direction = zeros(n_operations)
        direction[operation] = 1.0
        add_linear_constraint!(
            model,
            direction,
            :<=,
            1.0 - normalized_durations[operation],
        )
    end

    for job in 1:config.n_jobs, position in 1:(config.n_machines - 1)
        current = _operation(config, job, position)
        following = _operation(config, job, position + 1)
        direction = zeros(n_operations)
        direction[following] = 1.0
        direction[current] = -1.0
        add_linear_constraint!(
            model,
            direction,
            :>=,
            normalized_durations[current],
        )
    end

    for machine in 1:config.n_machines
        operations = findall(==(machine), machines)
        length(operations) == config.n_jobs || throw(ArgumentError(
            "Every job must visit every machine exactly once.",
        ))
        for left_index in 1:(length(operations) - 1)
            for right_index in (left_index + 1):length(operations)
                left, right = operations[left_index], operations[right_index]
                left_before = zeros(n_operations)
                left_before[left] = 1.0
                left_before[right] = -1.0
                right_before = -left_before
                add_disjunction!(
                    model,
                    [LinearConstraint(
                        left_before, :<=, -normalized_durations[left],
                    )],
                    [LinearConstraint(
                        right_before, :<=, -normalized_durations[right],
                    )];
                    name = Symbol("machine_$(machine)_$(left)_$(right)"),
                )
            end
        end
    end
    return model
end

_makespan(starts, durations) = maximum(starts .+ durations)

function _serial_dispatch(raw_durations, config)
    starts = zeros(Float64, length(raw_durations))
    cursor = 0.0
    for job in 1:config.n_jobs, position in 1:config.n_machines
        operation = _operation(config, job, position)
        starts[operation] = cursor
        cursor += raw_durations[operation]
    end
    return starts
end

function _milp_oracle(raw_durations, machines, config)
    n_operations = length(raw_durations)
    horizon = sum(raw_durations)
    jump_model = Model(HiGHS.Optimizer)
    set_silent(jump_model)
    @variable(jump_model, 0 <= starts[1:n_operations] <= horizon)
    @variable(jump_model, 0 <= completion <= horizon)

    for operation in 1:n_operations
        @constraint(
            jump_model,
            starts[operation] + raw_durations[operation] <= completion,
        )
    end
    for job in 1:config.n_jobs, position in 1:(config.n_machines - 1)
        current = _operation(config, job, position)
        following = _operation(config, job, position + 1)
        @constraint(
            jump_model,
            starts[following] >= starts[current] + raw_durations[current],
        )
    end
    for machine in 1:config.n_machines
        operations = findall(==(machine), machines)
        for left_index in 1:(length(operations) - 1)
            for right_index in (left_index + 1):length(operations)
                left, right = operations[left_index], operations[right_index]
                order = @variable(jump_model, binary = true)
                @constraint(
                    jump_model,
                    starts[left] + raw_durations[left] <=
                        starts[right] + horizon * (1 - order),
                )
                @constraint(
                    jump_model,
                    starts[right] + raw_durations[right] <=
                        starts[left] + horizon * order,
                )
            end
        end
    end
    @objective(jump_model, Min, completion)
    optimize!(jump_model)
    termination_status(jump_model) == MOI.OPTIMAL || error(
        "Job-shop makespan oracle did not return OPTIMAL.",
    )
    optimal_completion = objective_value(jump_model)

    @constraint(jump_model, completion <= optimal_completion + 1e-7)
    weights = 1.0 .+ collect(1:n_operations) ./ (100.0n_operations)
    @objective(jump_model, Min, sum(weights[i] * starts[i] for i in 1:n_operations))
    optimize!(jump_model)
    termination_status(jump_model) == MOI.OPTIMAL || error(
        "Job-shop canonical oracle tie-break did not return OPTIMAL.",
    )
    canonical_starts = Float64.(value.(starts))
    return canonical_starts, optimal_completion
end

function _raw_input(raw_durations, machines, config)
    n_operations = length(raw_durations)
    features = zeros(Float64, n_operations + n_operations * config.n_machines)
    duration_scale = max(config.train_duration_max, config.ood_duration_max)
    features[1:n_operations] .= raw_durations ./ duration_scale
    offset = n_operations
    for operation in 1:n_operations
        features[offset + (operation - 1) * config.n_machines + machines[operation]] = 1.0
    end
    return features
end

function _sample_instance(rng, distribution, config)
    duration_range = distribution in (:train, :iid) ?
        (config.train_duration_min:config.train_duration_max) :
        (config.ood_duration_min:config.ood_duration_max)
    n_operations = config.n_jobs * config.n_machines
    raw_durations = rand(rng, duration_range, n_operations)
    machines = Int[]
    for _ in 1:config.n_jobs
        append!(machines, randperm(rng, config.n_machines))
    end
    horizon = sum(raw_durations)
    normalized_durations = Float64.(raw_durations) ./ horizon
    model = build_jobshop_model(normalized_durations, machines, config)
    oracle_raw, oracle_completion = _milp_oracle(raw_durations, machines, config)
    oracle_starts = oracle_raw ./ horizon
    oracle = lazy_branch_search(
        model,
        oracle_starts;
        store_prune_records = false,
    )
    oracle.status == :optimal || error("Job-shop oracle branch encoding was unresolved.")
    oracle.objective <= 1e-10 || error("MILP oracle was not feasible in branch encoding.")
    oracle.report !== nothing && oracle.report.feasible ||
        error("Job-shop oracle failed original logical validation.")
    heuristic_raw = _serial_dispatch(raw_durations, config)
    heuristic_starts = heuristic_raw ./ horizon
    check_feasibility(model, heuristic_starts).feasible ||
        error("Serial job-shop initialization is not feasible.")
    return JobShopInstance(
        normalized_durations,
        raw_durations,
        machines,
        horizon,
        model,
        oracle_starts,
        copy(oracle.choices),
        oracle_completion / horizon,
        heuristic_starts,
        _makespan(heuristic_starts, normalized_durations),
    )
end

function _raw_split(config, count, seed, distribution)
    count > 0 || throw(ArgumentError("Job-shop split size must be positive."))
    rng = MersenneTwister(seed)
    instances = JobShopInstance[]
    raw_inputs = Matrix{Float64}(undef, 36, count)
    targets = Matrix{Float64}(undef, 9, count)
    for index in 1:count
        instance = _sample_instance(rng, distribution, config)
        push!(instances, instance)
        raw_inputs[:, index] .= _raw_input(
            instance.raw_durations, instance.machines, config,
        )
        targets[:, index] .= instance.oracle_starts
    end
    return (raw_inputs = raw_inputs, targets = targets, instances = instances)
end

function _normalized_split(raw, center, scale)
    return JobShopSplit(
        raw.raw_inputs,
        (raw.raw_inputs .- center) ./ scale,
        raw.targets,
        raw.instances,
    )
end

function prepare_problem(
    seed::Integer;
    config::JobShopConfig = JobShopConfig(),
    train_size::Int = 60,
    iid_size::Int = 80,
    ood_size::Int = 80,
)
    _validate_config(config)
    train_raw = _raw_split(config, train_size, 1_010_000 + Int(seed), :train)
    iid_raw = _raw_split(config, iid_size, 1_020_002, :iid)
    ood_raw = _raw_split(config, ood_size, 1_030_003, :ood)
    center = vec(mean(train_raw.raw_inputs; dims = 2))
    scale = vec(std(train_raw.raw_inputs; dims = 2, corrected = true))
    # A route category can be absent from a tiny smoke split. A fixed floor on
    # [0,1] features prevents an unseen one-hot value from being amplified by
    # six orders of magnitude without using test statistics.
    scale .= max.(scale, 0.1)
    return JobShopProblem(
        config,
        _normalized_split(train_raw, center, scale),
        _normalized_split(iid_raw, center, scale),
        _normalized_split(ood_raw, center, scale),
        center,
        scale,
    )
end

function initialize_network(
    seed::Integer;
    input_dimension::Int = 36,
    hidden::Int = 32,
    output_dimension::Int = 9,
)
    rng = MersenneTwister(1_040_000 + Int(seed))
    glorot(rows, columns) =
        randn(rng, rows, columns) .* sqrt(2.0 / (rows + columns))
    return JobShopMLP(
        glorot(hidden, input_dimension), zeros(hidden),
        glorot(hidden, hidden), zeros(hidden),
        glorot(output_dimension, hidden), zeros(output_dimension),
    )
end

_parameters(network::JobShopMLP) =
    [network.W1, network.b1, network.W2, network.b2, network.W3, network.b3]

function _forward(network::JobShopMLP, input)
    first_pre = network.W1 * input .+ network.b1
    first = max.(first_pre, 0.0)
    second_pre = network.W2 * first .+ network.b2
    second = max.(second_pre, 0.0)
    output = network.W3 * second .+ network.b3
    return output, (input, first_pre, first, second_pre, second)
end

function _zero_gradients(network)
    return [zeros(size(parameter)) for parameter in _parameters(network)]
end

function _backward!(gradients, network, cache, output_gradient)
    input, first_pre, first, second_pre, second = cache
    dW1, db1, dW2, db2, dW3, db3 = gradients
    dW3 .+= output_gradient * transpose(second)
    db3 .+= output_gradient
    second_gradient = transpose(network.W3) * output_gradient
    second_gradient .*= second_pre .> 0.0
    dW2 .+= second_gradient * transpose(first)
    db2 .+= second_gradient
    first_gradient = transpose(network.W2) * second_gradient
    first_gradient .*= first_pre .> 0.0
    dW1 .+= first_gradient * transpose(input)
    db1 .+= first_gradient
    return nothing
end

function _adam_state(network)
    parameters = _parameters(network)
    return AdamState(
        [zeros(size(parameter)) for parameter in parameters],
        [zeros(size(parameter)) for parameter in parameters],
        0,
    )
end

function _adam_update!(network, gradients, state, learning_rate)
    state.step += 1
    parameters = _parameters(network)
    for index in eachindex(parameters)
        first, second, gradient = state.first[index], state.second[index], gradients[index]
        first .= 0.9 .* first .+ 0.1 .* gradient
        second .= 0.999 .* second .+ 0.001 .* gradient .^ 2
        first_hat = first ./ (1.0 - 0.9^state.step)
        second_hat = second ./ (1.0 - 0.999^state.step)
        parameters[index] .-= learning_rate .* first_hat ./ (sqrt.(second_hat) .+ 1e-8)
    end
    return nothing
end

function _task_loss(output, instance, config)
    mse = sum(abs2, output .- instance.oracle_starts) / length(output)
    gap = _makespan(output, instance.durations) - instance.oracle_makespan
    return mse + config.makespan_loss_weight * gap^2
end

function _task_loss_gradient(output, instance, config)
    gradient = 2.0 .* (output .- instance.oracle_starts) ./ length(output)
    ends = output .+ instance.durations
    active = argmax(ends)
    gap = ends[active] - instance.oracle_makespan
    gradient[active] += 2.0 * config.makespan_loss_weight * gap
    return gradient
end

function pretrain_network(
    problem::JobShopProblem,
    seed::Integer;
    epochs::Int = 20,
    batch_size::Int = 16,
    learning_rate::Real = 2e-3,
)
    epochs > 0 || throw(ArgumentError("pretraining epochs must be positive."))
    batch_size > 0 || throw(ArgumentError("pretraining batch size must be positive."))
    network = initialize_network(seed; hidden = problem.config.hidden)
    state = _adam_state(network)
    rng = MersenneTwister(1_050_000 + Int(seed))
    count = size(problem.train.inputs, 2)
    for _ in 1:epochs
        order = randperm(rng, count)
        for start in 1:batch_size:count
            members = order[start:min(start + batch_size - 1, count)]
            gradients = _zero_gradients(network)
            for index in members
                output, cache = _forward(network, problem.train.inputs[:, index])
                target = problem.train.instances[index].heuristic_starts
                output_gradient = 2.0 .* (output .- target) ./ length(target)
                _backward!(gradients, network, cache, output_gradient)
            end
            for gradient in gradients
                gradient ./= length(members)
            end
            _adam_update!(network, gradients, state, Float64(learning_rate))
        end
    end
    return network
end

function _record_center!(counters, center)
    counters.center_global_searches += 1
    counters.center_leaf_solves += center.leaf_solves
    counters.center_relaxation_solves += center.relaxation_solves
    counters.logical_failures += center.report === nothing || !center.report.feasible
    return nothing
end

function _gradient_for_sample(
    method,
    instance,
    point,
    perturbations,
    radius,
    config,
    counters,
    rng,
)
    center = lazy_branch_search(
        instance.model, point; store_prune_records = false,
    )
    center.status == :optimal || error(
        "Job-shop center projection unresolved: status=$(center.status), " *
        "unresolved_leaves=$(center.unresolved_leaves).",
    )
    _record_center!(counters, center)
    center_loss = _task_loss(center.y, instance, config)

    if method == :hard_local
        branch = project_branch(instance.model, point, center.choices)
        local_gradient = fixed_branch_loss_gradient(
            instance.model,
            point,
            center.choices,
            y -> _task_loss_gradient(y, instance, config);
            branch_result = branch,
        )
        counters.reference_branch_solves += 1
        local_gradient.status == :ok || error("Job-shop hard-local gradient unresolved.")
        return local_gradient.gradient, center_loss
    elseif method == :ste
        return _task_loss_gradient(center.y, instance, config), center_loss
    elseif method == :blackbox
        total = zeros(length(point))
        for column in axes(perturbations, 2)
            direction = perturbations[:, column]
            result = lazy_branch_search(
                instance.model,
                point .+ radius .* direction;
                store_prune_records = false,
            )
            result.status == :optimal || error("Job-shop black-box search unresolved.")
            counters.perturbed_global_searches += 1
            counters.lazy_leaf_solves += result.leaf_solves
            counters.lazy_relaxation_solves += result.relaxation_solves
            counters.switching_samples += result.choices != center.choices
            counters.logical_failures += result.report === nothing || !result.report.feasible
            residual = _task_loss(result.y, instance, config) - center_loss
            total .+= (residual / radius) .* ball_smoothing_score(direction; beta = 2.0)
        end
        return total ./ size(perturbations, 2), center_loss
    elseif method in (:sage_cert, :sage_geometry, :sage_multiscale)
        estimate = if method == :sage_cert
            sage_gradient(
                instance.model,
                point,
                radius,
                y -> _task_loss(y, instance, config),
                y -> _task_loss_gradient(y, instance, config);
                samples = perturbations,
                center_result = center,
                store_records = false,
            )
        elseif method == :sage_geometry
            sage_config = SAGEConfig(
                radius = radius,
                n_samples = size(perturbations, 2),
                beta = 2.0,
                sampling = :geometry,
                store_records = false,
            )
            sage_gradient(
                instance.model,
                point,
                y -> _task_loss(y, instance, config),
                y -> _task_loss_gradient(y, instance, config),
                sage_config;
                rng = rng,
                center_result = center,
            )
        else
            boundary_certificate = switching_certificate(
                instance.model,
                point,
                radius;
                center_result = center,
            )
            relevant_groups = competitor_set(boundary_certificate)
            geometry = switching_geometry(
                instance.model,
                point,
                boundary_certificate.winner,
                boundary_certificate.winner_point,
                relevant_groups,
            )
            local_radius = certified_local_radius(boundary_certificate)
            counters.local_radius_sum += local_radius
            counters.local_radius_count += 1
            counters.positive_local_radius_calls += local_radius > 0.0
            counters.geometry_branch_solves += geometry.branch_solves
            if size(geometry.normals, 2) == 0
                sage_gradient(
                    instance.model,
                    point,
                    radius,
                    y -> _task_loss(y, instance, config),
                    y -> _task_loss_gradient(y, instance, config);
                    samples = perturbations,
                    center_result = center,
                    store_records = false,
                )
            else
                has_local = local_radius > 0.0
                local_weight = has_local ? 0.4 : 0.0
                base_weight = has_local ? 0.2 : 0.6
                batch = sample_certified_geometry_smoothing(
                    rng,
                    length(point),
                    size(perturbations, 2),
                    geometry.normals;
                    beta = 2.0,
                    base_weight = base_weight,
                    local_weight = local_weight,
                    geometry_weight = 0.4,
                    local_scale = has_local ? local_radius / radius : 1.0,
                    focus_strength = 0.9,
                )
                counters.base_proposal_samples += count(==(:base), batch.components)
                counters.local_proposal_samples += count(==(:local), batch.components)
                counters.focused_proposal_samples += count(==(:focused), batch.components)
                sage_gradient(
                    instance.model,
                    point,
                    radius,
                    y -> _task_loss(y, instance, config),
                    y -> _task_loss_gradient(y, instance, config);
                    samples = batch.samples,
                    importance_weights = batch.importance_weights,
                    sampling = :provided,
                    center_result = center,
                    store_records = false,
                )
            end
        end
        estimate.status == :ok || error(
            "Job-shop SAGE gradient unresolved: status=$(estimate.status), " *
            "geometry_status=$(estimate.geometry_status), " *
            "geometry_normals=$(estimate.geometry_normals), " *
            "unresolved_samples=$(estimate.unresolved_samples), " *
            "global_searches=$(estimate.perturbed_global_searches), " *
            "lazy_searches=$(estimate.lazy_global_searches).",
        )
        counters.perturbed_global_searches += estimate.perturbed_global_searches
        counters.reference_branch_solves += estimate.reference_branch_solves
        counters.competitor_branch_solves += estimate.competitor_branch_solves
        counters.geometry_branch_solves += estimate.geometry_branch_solves
        counters.sample_certificates += estimate.sample_stability_certificates
        counters.lazy_leaf_solves += estimate.lazy_leaf_solves
        counters.lazy_relaxation_solves += estimate.lazy_relaxation_solves
        counters.switching_samples += estimate.switching_samples
        counters.unresolved_samples += estimate.unresolved_samples
        counters.effective_sample_size_sum += estimate.importance_effective_sample_size
        counters.effective_sample_size_count += 1
        return estimate.gradient, center_loss
    end
    throw(ArgumentError("Unknown job-shop training method $(method)."))
end

function evaluate_network(
    network,
    split::JobShopSplit;
    constrained::Bool = true,
    config::JobShopConfig = JobShopConfig(),
)
    count = size(split.inputs, 2)
    task_loss = 0.0
    start_mse = 0.0
    makespan_gap = 0.0
    feasible = 0
    order_matches = 0
    order_total = 0
    exact_order_matches = 0
    unresolved = 0
    maximum_violation = 0.0
    for index in 1:count
        instance = split.instances[index]
        point, _ = _forward(network, split.inputs[:, index])
        if constrained
            result = lazy_branch_search(
                instance.model, point; store_prune_records = false,
            )
            if result.status != :optimal
                unresolved += 1
                continue
            end
            output, report = result.y, result.report
            order_matches += sum(result.choices .== instance.oracle_choices)
            order_total += length(result.choices)
            exact_order_matches += result.choices == instance.oracle_choices
        else
            output = point
            report = check_feasibility(instance.model, output)
        end
        task_loss += _task_loss(output, instance, config)
        start_mse += sum(abs2, output .- instance.oracle_starts) / length(output)
        predicted_makespan = _makespan(output, instance.durations)
        makespan_gap += max(0.0, predicted_makespan - instance.oracle_makespan) /
            instance.oracle_makespan
        feasible += report !== nothing && report.feasible
        maximum_violation = max(
            maximum_violation,
            report === nothing ? Inf : report.max_violation,
        )
    end
    resolved = count - unresolved
    return (
        task_loss = resolved > 0 ? task_loss / resolved : Inf,
        start_mse = resolved > 0 ? start_mse / resolved : Inf,
        makespan_gap = resolved > 0 ? makespan_gap / resolved : Inf,
        csat = feasible / count,
        order_agreement = constrained && order_total > 0 ? order_matches / order_total : NaN,
        exact_order_agreement = constrained ? exact_order_matches / count : NaN,
        maximum_violation = maximum_violation,
        unresolved = unresolved,
    )
end

function train_method(
    problem::JobShopProblem,
    pretrained::JobShopMLP,
    method::Symbol,
    seed::Integer;
    epochs::Int = 1,
    samples::Int = 8,
    radius::Real = 0.10,
    learning_rate::Real = 1e-3,
    evaluation_interval::Int = 20,
    max_steps::Int = typemax(Int),
)
    method in (
        :hard_local, :ste, :blackbox, :sage_cert, :sage_geometry,
        :sage_multiscale,
    ) ||
        throw(ArgumentError("Unsupported job-shop method $(method)."))
    samples > 0 && iseven(samples) ||
        throw(ArgumentError("samples must be a positive even integer."))
    network = deepcopy(pretrained)
    state = _adam_state(network)
    counters = TrainingCounters()
    trajectory = NamedTuple[]
    rng = MersenneTwister(1_060_000 + Int(seed))
    count = size(problem.train.inputs, 2)
    orders = [randperm(rng, count) for _ in 1:epochs]
    sequence = reduce(vcat, orders)[1:min(max_steps, epochs * count)]

    function record!(step)
        iid = evaluate_network(network, problem.iid; config = problem.config)
        ood = evaluate_network(network, problem.ood; config = problem.config)
        push!(trajectory, (
            step = step,
            iid_task_loss = iid.task_loss,
            iid_makespan_gap = iid.makespan_gap,
            ood_task_loss = ood.task_loss,
            ood_makespan_gap = ood.makespan_gap,
        ))
    end

    record!(0)
    started = time_ns()
    for (step, index) in enumerate(sequence)
        gradients = _zero_gradients(network)
        point, cache = _forward(network, problem.train.inputs[:, index])
        perturbation_rng = MersenneTwister(
            1_070_000_000 + 100_000Int(seed) + 1_000step + index,
        )
        perturbations = method in (
            :blackbox, :sage_cert, :sage_geometry, :sage_multiscale,
        ) ?
            sample_antithetic_ball_smoothing(
                perturbation_rng, length(point), samples; beta = 2.0,
            ) : zeros(length(point), samples)
        output_gradient, _ = try
            _gradient_for_sample(
                method,
                problem.train.instances[index],
                point,
                perturbations,
                Float64(radius),
                problem.config,
                counters,
                perturbation_rng,
            )
        catch error_value
            error(
                "Phase 10 training failure at method=$(method), seed=$(seed), " *
                "step=$(step), sample=$(index): " * sprint(showerror, error_value),
            )
        end
        all(isfinite, output_gradient) || error("Job-shop gradient became non-finite.")
        _backward!(gradients, network, cache, output_gradient)
        _adam_update!(network, gradients, state, Float64(learning_rate))
        if step % evaluation_interval == 0 || step == length(sequence)
            record!(step)
        end
    end
    training_seconds = (time_ns() - started) / 1e9
    mean_ess = counters.effective_sample_size_count > 0 ?
        counters.effective_sample_size_sum / counters.effective_sample_size_count : NaN
    mean_local_radius = counters.local_radius_count > 0 ?
        counters.local_radius_sum / counters.local_radius_count : NaN
    return (
        network = network,
        trajectory = trajectory,
        training_seconds = training_seconds,
        steps = length(sequence),
        center_global_searches = counters.center_global_searches,
        center_leaf_solves = counters.center_leaf_solves,
        center_relaxation_solves = counters.center_relaxation_solves,
        perturbed_global_searches = counters.perturbed_global_searches,
        reference_branch_solves = counters.reference_branch_solves,
        competitor_branch_solves = counters.competitor_branch_solves,
        geometry_branch_solves = counters.geometry_branch_solves,
        sample_certificates = counters.sample_certificates,
        lazy_leaf_solves = counters.lazy_leaf_solves,
        lazy_relaxation_solves = counters.lazy_relaxation_solves,
        switching_samples = counters.switching_samples,
        unresolved_samples = counters.unresolved_samples,
        logical_failures = counters.logical_failures,
        complete_branch_solves = counters.center_leaf_solves +
            counters.reference_branch_solves + counters.competitor_branch_solves +
            counters.geometry_branch_solves + counters.lazy_leaf_solves,
        mean_effective_sample_size = mean_ess,
        mean_local_radius = mean_local_radius,
        positive_local_radius_calls = counters.positive_local_radius_calls,
        base_proposal_samples = counters.base_proposal_samples,
        local_proposal_samples = counters.local_proposal_samples,
        focused_proposal_samples = counters.focused_proposal_samples,
    )
end

end # module
