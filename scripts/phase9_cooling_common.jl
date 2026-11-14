module Phase9Cooling

using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

export CoolingConfig
export CoolingMLP
export CoolingProblem
export CoolingSplit
export active_rule_names
export build_cooling_model
export evaluate_network
export initialize_network
export prepare_problem
export pretrain_network
export train_method

"""All declared choices used by the paper-spec cooling reconstruction."""
Base.@kwdef struct CoolingConfig
    hot_threshold::Float64 = 30.0
    cold_threshold::Float64 = 15.0
    humid_threshold::Float64 = 70.0
    high_workload_threshold::Float64 = 0.70
    fan_load_coefficient::Float64 = 0.55
    chiller_load_coefficient::Float64 = 0.75
    base_load::Float64 = 0.10
    workload_load::Float64 = 0.30
    temperature_load::Float64 = 0.010
    power_fan::Float64 = 0.35
    power_chiller::Float64 = 0.50
    power_pump::Float64 = 0.15
    demand_power_cap::Float64 = 0.55
    pump_minimum::Float64 = 0.10
    oracle_energy_weight::Float64 = 0.08
    target_noise::Float64 = 0.01
end

struct CoolingSplit
    raw_inputs::Matrix{Float64}
    inputs::Matrix{Float64}
    targets::Matrix{Float64}
    oracle_outputs::Matrix{Float64}
    oracle_choices::Vector{Vector{Int}}
    active_rules::Vector{Vector{Symbol}}
    models::Vector{DisjunctiveModel}
end

struct CoolingProblem
    config::CoolingConfig
    train::CoolingSplit
    iid::CoolingSplit
    ood::CoolingSplit
    input_mean::Vector{Float64}
    input_scale::Vector{Float64}
end

mutable struct CoolingMLP
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
end

TrainingCounters() = TrainingCounters(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0,
)

function active_rule_names(input, config::CoolingConfig = CoolingConfig())
    length(input) == 7 || throw(DimensionMismatch("Cooling inputs have seven coordinates."))
    temperature, humidity, workload, _, demand, fan_maintenance, chiller_maintenance = input
    rules = Symbol[]
    temperature > config.hot_threshold && push!(rules, :hot)
    temperature < config.cold_threshold && push!(rules, :cold)
    humidity > config.humid_threshold && push!(rules, :humid)
    workload > config.high_workload_threshold && push!(rules, :high_workload)
    demand >= 0.5 && push!(rules, :demand_response)
    fan_maintenance >= 0.5 && push!(rules, :fan_maintenance)
    chiller_maintenance >= 0.5 && push!(rules, :chiller_maintenance)
    return rules
end

function _constraint(coordinate, sense, bound)
    direction = zeros(3)
    direction[coordinate] = 1.0
    return LinearConstraint(direction, sense, bound)
end

function build_cooling_model(input, config::CoolingConfig = CoolingConfig())
    length(input) == 7 || throw(DimensionMismatch("Cooling inputs have seven coordinates."))
    temperature, _, workload = input[1], input[2], input[3]
    model = DisjunctiveModel(3)
    set_bounds!(model; lower = zeros(3), upper = ones(3))

    required_load = config.base_load + config.workload_load * workload +
        config.temperature_load * max(0.0, temperature - 20.0)
    add_linear_constraint!(
        model,
        [config.fan_load_coefficient, config.chiller_load_coefficient, 0.0],
        :>=,
        required_load,
    )
    add_linear_constraint!(model, [0.0, 0.0, 1.0], :>=, config.pump_minimum)

    for rule in active_rule_names(input, config)
        if rule == :hot
            add_disjunction!(
                model,
                [_constraint(2, :>=, 0.45)],
                [_constraint(1, :>=, 0.75)];
                name = rule,
            )
        elseif rule == :cold
            add_disjunction!(
                model,
                [_constraint(2, :<=, 0.05)],
                [_constraint(1, :>=, 0.60)];
                name = rule,
            )
        elseif rule == :humid
            add_disjunction!(
                model,
                [_constraint(1, :<=, 0.70)],
                [_constraint(2, :>=, 0.30)];
                name = rule,
            )
        elseif rule == :high_workload
            add_disjunction!(
                model,
                [_constraint(1, :>=, 0.65)],
                [_constraint(2, :>=, 0.35)];
                name = rule,
            )
        elseif rule == :demand_response
            power = [config.power_fan, config.power_chiller, config.power_pump]
            add_disjunction!(
                model,
                [LinearConstraint(power, :<=, config.demand_power_cap)],
                [_constraint(1, :<=, 0.50), _constraint(2, :<=, 0.40)];
                name = rule,
            )
        elseif rule == :fan_maintenance
            add_disjunction!(
                model,
                [_constraint(1, :<=, 0.40)],
                [_constraint(2, :>=, 0.40)];
                name = rule,
            )
        elseif rule == :chiller_maintenance
            add_disjunction!(
                model,
                [_constraint(2, :<=, 0.40)],
                [_constraint(1, :>=, 0.70)];
                name = rule,
            )
        else
            error("Unknown cooling rule $(rule).")
        end
    end
    return model
end

function _reference_setpoint(input)
    temperature, humidity, workload = input[1], input[2], input[3]
    hot = max(0.0, temperature - 20.0)
    humid = max(0.0, humidity - 50.0)
    fan = clamp(0.15 + 0.55workload + 0.008hot, 0.0, 1.0)
    chiller = clamp(0.05 + 0.35workload + 0.012hot + 0.002humid, 0.0, 1.0)
    pump = clamp(0.12 + 0.18workload, 0.0, 1.0)
    return [fan, chiller, pump]
end

function _oracle_point(input, config)
    reference = _reference_setpoint(input)
    price = input[4]
    power = [config.power_fan, config.power_chiller, config.power_pump]
    # Completing the square shows that projecting this shifted point minimizes
    # 0.5||y-reference||^2 + energy_weight * price * power' * y.
    return reference .- config.oracle_energy_weight .* price .* power
end

function _sample_input(rng, distribution::Symbol)
    if distribution in (:train, :iid)
        temperature = 40.0rand(rng)
        humidity = 10.0 + 85.0rand(rng)
        workload = rand(rng)
        price = 0.05 + 0.45rand(rng)
        probabilities = (0.10, 0.05, 0.05)
    elseif distribution == :ood
        temperature = 18.0 + 27.0rand(rng)
        humidity = 50.0 + 50.0rand(rng)
        workload = 0.4 + 0.6rand(rng)
        price = 0.20 + 0.60rand(rng)
        probabilities = (0.35, 0.08, 0.08)
    else
        throw(ArgumentError("Unknown cooling distribution $(distribution)."))
    end
    events = Float64[rand(rng) < probability for probability in probabilities]
    return [temperature, humidity, workload, price, events...]
end

function _raw_split(config, count, seed, distribution)
    count > 0 || throw(ArgumentError("Cooling split size must be positive."))
    rng = MersenneTwister(seed)
    raw_inputs = Matrix{Float64}(undef, 7, count)
    targets = Matrix{Float64}(undef, 3, count)
    oracle_outputs = Matrix{Float64}(undef, 3, count)
    choices = Vector{Vector{Int}}(undef, count)
    active_rules = Vector{Vector{Symbol}}(undef, count)
    models = Vector{DisjunctiveModel}(undef, count)
    for index in 1:count
        input = _sample_input(rng, distribution)
        model = build_cooling_model(input, config)
        oracle = lazy_branch_search(
            model,
            _oracle_point(input, config);
            store_prune_records = false,
        )
        oracle.status == :optimal || error("Cooling oracle was not globally resolved.")
        oracle.report !== nothing && oracle.report.feasible ||
            error("Cooling oracle returned an infeasible point.")
        raw_inputs[:, index] .= input
        oracle_outputs[:, index] .= oracle.y
        targets[:, index] .= clamp.(
            oracle.y .+ config.target_noise .* randn(rng, 3), 0.0, 1.0,
        )
        choices[index] = copy(oracle.choices)
        active_rules[index] = active_rule_names(input, config)
        models[index] = model
    end
    return (
        raw_inputs = raw_inputs,
        targets = targets,
        oracle_outputs = oracle_outputs,
        oracle_choices = choices,
        active_rules = active_rules,
        models = models,
    )
end

function _normalized_split(raw, center, scale)
    inputs = (raw.raw_inputs .- center) ./ scale
    return CoolingSplit(
        raw.raw_inputs,
        inputs,
        raw.targets,
        raw.oracle_outputs,
        raw.oracle_choices,
        raw.active_rules,
        raw.models,
    )
end

function prepare_problem(
    seed::Integer;
    config::CoolingConfig = CoolingConfig(),
    train_size::Int = 250,
    iid_size::Int = 2000,
    ood_size::Int = 2000,
)
    train_raw = _raw_split(config, train_size, 900000 + Int(seed), :train)
    iid_raw = _raw_split(config, iid_size, 2, :iid)
    ood_raw = _raw_split(config, ood_size, 3, :ood)
    center = vec(mean(train_raw.raw_inputs; dims = 2))
    scale = vec(std(train_raw.raw_inputs; dims = 2, corrected = true))
    # Rare binary events can be absent in tiny smoke splits. Keep a finite,
    # train-only normalization without leaking test statistics.
    scale .= max.(scale, 1.0)
    return CoolingProblem(
        config,
        _normalized_split(train_raw, center, scale),
        _normalized_split(iid_raw, center, scale),
        _normalized_split(ood_raw, center, scale),
        center,
        scale,
    )
end

function initialize_network(seed::Integer; hidden::Int = 8)
    hidden > 0 || throw(ArgumentError("hidden width must be positive."))
    rng = MersenneTwister(910000 + Int(seed))
    return CoolingMLP(
        sqrt(2 / 7) .* randn(rng, hidden, 7),
        zeros(hidden),
        sqrt(2 / hidden) .* randn(rng, hidden, hidden),
        zeros(hidden),
        0.25 .* randn(rng, 3, hidden),
        zeros(3),
    )
end

_parameters(network) = (
    network.W1, network.b1, network.W2, network.b2, network.W3, network.b3,
)

function _forward(network::CoolingMLP, input)
    pre1 = network.W1 * input .+ network.b1
    hidden1 = max.(pre1, 0.0)
    pre2 = network.W2 * hidden1 .+ network.b2
    hidden2 = max.(pre2, 0.0)
    output = network.W3 * hidden2 .+ network.b3
    return output, (input, pre1, hidden1, pre2, hidden2)
end

function _zero_gradients(network)
    return [zeros(size(parameter)) for parameter in _parameters(network)]
end

function _backward!(gradients, network, cache, output_gradient)
    input, pre1, hidden1, pre2, hidden2 = cache
    gradients[5] .+= output_gradient * hidden2'
    gradients[6] .+= output_gradient
    hidden2_gradient = (network.W3' * output_gradient) .* (pre2 .> 0.0)
    gradients[3] .+= hidden2_gradient * hidden1'
    gradients[4] .+= hidden2_gradient
    hidden1_gradient = (network.W2' * hidden2_gradient) .* (pre1 .> 0.0)
    gradients[1] .+= hidden1_gradient * input'
    gradients[2] .+= hidden1_gradient
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

_loss(y, target) = sum(abs2, y .- target) / length(target)
_loss_gradient(y, target) = 2.0 .* (y .- target) ./ length(target)

function pretrain_network(
    problem::CoolingProblem,
    seed::Integer;
    epochs::Int = 5,
    batch_size::Int = 64,
    learning_rate::Real = 1e-3,
    hidden::Int = 8,
)
    epochs > 0 || throw(ArgumentError("pretraining epochs must be positive."))
    batch_size > 0 || throw(ArgumentError("pretraining batch size must be positive."))
    network = initialize_network(seed; hidden = hidden)
    state = _adam_state(network)
    rng = MersenneTwister(920000 + Int(seed))
    count = size(problem.train.inputs, 2)
    for _ in 1:epochs
        order = randperm(rng, count)
        for start in 1:batch_size:count
            members = order[start:min(start + batch_size - 1, count)]
            gradients = _zero_gradients(network)
            for index in members
                output, cache = _forward(network, problem.train.inputs[:, index])
                _backward!(
                    gradients,
                    network,
                    cache,
                    _loss_gradient(output, problem.train.targets[:, index]),
                )
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
    model,
    point,
    target,
    perturbations,
    radius,
    counters,
    rng,
)
    center = lazy_branch_search(model, point; store_prune_records = false)
    center.status == :optimal || error(
        "Cooling center projection was unresolved: status=$(center.status), " *
        "termination=$(center.termination_reason), unresolved_leaves=$(center.unresolved_leaves), " *
        "relaxation_failures=$(center.relaxation_failures), max_abs_point=$(maximum(abs, point)).",
    )
    _record_center!(counters, center)
    center_loss = _loss(center.y, target)

    if method == :hard_local
        branch = project_branch(model, point, center.choices)
        local_gradient = fixed_branch_loss_gradient(
            model,
            point,
            center.choices,
            y -> _loss_gradient(y, target);
            branch_result = branch,
        )
        counters.reference_branch_solves += 1
        local_gradient.status == :ok || error("Cooling hard-local gradient was unresolved.")
        return local_gradient.gradient, center_loss
    elseif method == :ste
        return _loss_gradient(center.y, target), center_loss
    elseif method == :blackbox
        total = zeros(3)
        for column in axes(perturbations, 2)
            u = perturbations[:, column]
            result = lazy_branch_search(
                model,
                point .+ radius .* u;
                store_prune_records = false,
            )
            result.status == :optimal || error("Cooling black-box search was unresolved.")
            counters.perturbed_global_searches += 1
            counters.lazy_leaf_solves += result.leaf_solves
            counters.lazy_relaxation_solves += result.relaxation_solves
            counters.switching_samples += result.choices != center.choices
            counters.logical_failures += result.report === nothing || !result.report.feasible
            residual = _loss(result.y, target) - center_loss
            total .+= (residual / radius) .* ball_smoothing_score(u; beta = 2.0)
        end
        return total ./ size(perturbations, 2), center_loss
    elseif method in (:sage_cert, :sage_geometry)
        estimate = if method == :sage_cert
            sage_gradient(
                model,
                point,
                radius,
                y -> _loss(y, target),
                y -> _loss_gradient(y, target);
                samples = perturbations,
                center_result = center,
                store_records = false,
            )
        else
            config = SAGEConfig(
                radius = radius,
                n_samples = size(perturbations, 2),
                beta = 2.0,
                sampling = :geometry,
                store_records = false,
            )
            sage_gradient(
                model,
                point,
                y -> _loss(y, target),
                y -> _loss_gradient(y, target),
                config;
                rng = rng,
                center_result = center,
            )
        end
        estimate.status == :ok || error("Cooling SAGE gradient was unresolved.")
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
    throw(ArgumentError("Unknown cooling training method $(method)."))
end

function evaluate_network(network, split::CoolingSplit; constrained::Bool = true)
    count = size(split.inputs, 2)
    count > 0 || throw(ArgumentError("Cooling evaluation split cannot be empty."))
    squared_error = 0.0
    feasible = 0
    agreement = 0
    maximum_violation = 0.0
    unresolved = 0
    for index in 1:count
        point, _ = _forward(network, split.inputs[:, index])
        if constrained
            result = lazy_branch_search(
                split.models[index], point; store_prune_records = false,
            )
            if result.status != :optimal
                unresolved += 1
                continue
            end
            prediction = result.y
            report = result.report
            agreement += result.choices == split.oracle_choices[index]
        else
            prediction = point
            report = check_feasibility(split.models[index], prediction)
        end
        squared_error += _loss(prediction, split.targets[:, index])
        feasible += report !== nothing && report.feasible
        maximum_violation = max(
            maximum_violation,
            report === nothing ? Inf : report.max_violation,
        )
    end
    resolved = count - unresolved
    return (
        mse = resolved > 0 ? squared_error / resolved : Inf,
        csat = feasible / count,
        branch_agreement = constrained ? agreement / count : NaN,
        maximum_violation = maximum_violation,
        unresolved = unresolved,
    )
end

function train_method(
    problem::CoolingProblem,
    pretrained::CoolingMLP,
    method::Symbol,
    seed::Integer;
    epochs::Int = 1,
    samples::Int = 8,
    radius::Real = 0.15,
    learning_rate::Real = 1e-3,
    evaluation_interval::Int = 50,
    max_steps::Int = typemax(Int),
)
    method in (:hard_local, :ste, :blackbox, :sage_cert, :sage_geometry) ||
        throw(ArgumentError("Unsupported cooling method $(method)."))
    epochs > 0 || throw(ArgumentError("fine-tuning epochs must be positive."))
    samples > 0 && iseven(samples) ||
        throw(ArgumentError("samples must be a positive even integer."))
    radius > 0 || throw(ArgumentError("radius must be positive."))
    evaluation_interval > 0 || throw(ArgumentError("evaluation interval must be positive."))
    network = deepcopy(pretrained)
    state = _adam_state(network)
    counters = TrainingCounters()
    trajectory = NamedTuple[]
    rng = MersenneTwister(930000 + Int(seed))
    count = size(problem.train.inputs, 2)
    orders = [randperm(rng, count) for _ in 1:epochs]
    sequence = reduce(vcat, orders)[1:min(max_steps, epochs * count)]

    function record!(step)
        iid_metrics = evaluate_network(network, problem.iid; constrained = true)
        ood_metrics = evaluate_network(network, problem.ood; constrained = true)
        push!(trajectory, (
            step = step,
            iid_mse = iid_metrics.mse,
            iid_csat = iid_metrics.csat,
            ood_mse = ood_metrics.mse,
            ood_csat = ood_metrics.csat,
        ))
    end

    record!(0)
    started = time_ns()
    for (step, index) in enumerate(sequence)
        gradients = _zero_gradients(network)
        point, cache = _forward(network, problem.train.inputs[:, index])
        perturbation_rng = MersenneTwister(
            940000000 + 100000Int(seed) + 1000step + index,
        )
        perturbations = method in (:blackbox, :sage_cert, :sage_geometry) ?
            sample_antithetic_ball_smoothing(
                perturbation_rng, 3, samples; beta = 2.0,
            ) : zeros(3, samples)
        output_gradient, _ = try
            _gradient_for_sample(
                method,
                problem.train.models[index],
                point,
                problem.train.targets[:, index],
                perturbations,
                Float64(radius),
                counters,
                perturbation_rng,
            )
        catch error_value
            error(
                "Phase 9 training failure at method=$(method), seed=$(seed), " *
                "step=$(step), sample=$(index): " * sprint(showerror, error_value),
            )
        end
        all(isfinite, output_gradient) || error("Cooling gradient became non-finite.")
        _backward!(gradients, network, cache, output_gradient)
        _adam_update!(network, gradients, state, Float64(learning_rate))
        if step % evaluation_interval == 0 || step == length(sequence)
            record!(step)
        end
    end
    training_seconds = (time_ns() - started) / 1e9
    mean_ess = counters.effective_sample_size_count > 0 ?
        counters.effective_sample_size_sum / counters.effective_sample_size_count : NaN
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
        mean_effective_sample_size = mean_ess,
    )
end

end # module
