module Phase8Iris

using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

export IrisDataset
export IrisProblem
export TinyMLP
export analytic_project
export evaluate_network
export initialize_network
export load_iris
export prepare_problem
export train_method

const CLASS_NAMES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

struct IrisDataset
    inputs::Matrix{Float64}
    targets::Matrix{Float64}
    labels::Vector{Int}
end

struct IrisProblem
    inputs::Matrix{Float64}
    targets::Matrix{Float64}
    raw_targets::Matrix{Float64}
    labels::Vector{Int}
    train::Vector{Int}
    validation::Vector{Int}
    test::Vector{Int}
    input_mean::Vector{Float64}
    input_scale::Vector{Float64}
    target_mean::Vector{Float64}
    target_scale::Vector{Float64}
    branch_lower::Matrix{Float64}
    branch_upper::Matrix{Float64}
    model::DisjunctiveModel
end

mutable struct TinyMLP
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

mutable struct AdamState
    first_W1::Matrix{Float64}
    first_b1::Vector{Float64}
    first_W2::Matrix{Float64}
    first_b2::Vector{Float64}
    second_W1::Matrix{Float64}
    second_b1::Vector{Float64}
    second_W2::Matrix{Float64}
    second_b2::Vector{Float64}
    step::Int
end

mutable struct TrainingCounters
    center_global_searches::Int
    center_leaf_solves::Int
    center_relaxation_solves::Int
    perturbed_global_searches::Int
    reference_branch_solves::Int
    competitor_branch_solves::Int
    sample_stability_certificates::Int
    restricted_global_searches::Int
    lazy_global_searches::Int
    lazy_leaf_solves::Int
    lazy_relaxation_solves::Int
    switching_samples::Int
    unresolved_samples::Int
    logical_failures::Int
end

TrainingCounters() = TrainingCounters(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function load_iris(path::AbstractString)
    isfile(path) || throw(ArgumentError("Iris data file not found: $(path)"))
    inputs = Vector{Vector{Float64}}()
    targets = Vector{Vector{Float64}}()
    labels = Int[]
    for (line_number, raw_line) in enumerate(eachline(path))
        line = strip(raw_line)
        isempty(line) && continue
        fields = split(line, ',')
        length(fields) == 5 || error("Malformed Iris row $(line_number).")
        values = parse.(Float64, fields[1:4])
        label = findfirst(==(fields[5]), CLASS_NAMES)
        label === nothing && error("Unknown Iris class $(fields[5]).")
        push!(inputs, values[1:2])
        push!(targets, values[3:4])
        push!(labels, label)
    end
    length(labels) == 150 || error("Expected 150 Iris observations, found $(length(labels)).")
    all(count(==(label), labels) == 50 for label in 1:3) ||
        error("Iris class counts are not 50/50/50.")
    return IrisDataset(reduce(hcat, inputs), reduce(hcat, targets), labels)
end

function _stratified_split(labels, seed; train_per_class = 30, validation_per_class = 10)
    rng = MersenneTwister(810000 + seed)
    train = Int[]
    validation = Int[]
    test = Int[]
    for label in 1:3
        members = findall(==(label), labels)
        shuffled = members[randperm(rng, length(members))]
        train_end = train_per_class
        validation_end = train_end + validation_per_class
        append!(train, shuffled[1:train_end])
        append!(validation, shuffled[(train_end + 1):validation_end])
        append!(test, shuffled[(validation_end + 1):end])
    end
    shuffle!(rng, train)
    shuffle!(rng, validation)
    shuffle!(rng, test)
    return train, validation, test
end

function _normalization(matrix, indices)
    center = vec(mean(matrix[:, indices]; dims = 2))
    scale = vec(std(matrix[:, indices]; dims = 2, corrected = true))
    all(scale .> 1e-12) || error("A training feature has zero variance.")
    return center, scale
end

function _build_box_model(lower, upper)
    size(lower) == size(upper) || throw(DimensionMismatch("Box matrices must match."))
    dimension, branches = size(lower)
    model = DisjunctiveModel(dimension)
    set_bounds!(
        model;
        lower = vec(minimum(lower; dims = 2)),
        upper = vec(maximum(upper; dims = 2)),
    )
    disjuncts = Vector{LinearConstraint}[]
    for branch in 1:branches
        constraints = LinearConstraint[]
        for coordinate in 1:dimension
            direction = zeros(dimension)
            direction[coordinate] = 1.0
            push!(constraints, LinearConstraint(direction, :>=, lower[coordinate, branch]))
            push!(constraints, LinearConstraint(direction, :<=, upper[coordinate, branch]))
        end
        push!(disjuncts, constraints)
    end
    add_disjunction!(model, disjuncts...; name = :iris_species_box)
    return model
end

function prepare_problem(dataset::IrisDataset, seed; box_padding = 0.05)
    box_padding >= 0 || throw(ArgumentError("box_padding must be nonnegative."))
    train, validation, test = _stratified_split(dataset.labels, seed)
    input_mean, input_scale = _normalization(dataset.inputs, train)
    target_mean, target_scale = _normalization(dataset.targets, train)
    inputs = (dataset.inputs .- input_mean) ./ input_scale
    targets = (dataset.targets .- target_mean) ./ target_scale

    lower = zeros(2, 3)
    upper = zeros(2, 3)
    for label in 1:3
        members = [index for index in train if dataset.labels[index] == label]
        lower[:, label] .= vec(minimum(targets[:, members]; dims = 2)) .- box_padding
        upper[:, label] .= vec(maximum(targets[:, members]; dims = 2)) .+ box_padding
    end
    model = _build_box_model(lower, upper)
    return IrisProblem(
        inputs,
        targets,
        dataset.targets,
        copy(dataset.labels),
        train,
        validation,
        test,
        input_mean,
        input_scale,
        target_mean,
        target_scale,
        lower,
        upper,
        model,
    )
end

function analytic_project(problem::IrisProblem, point)
    length(point) == 2 || throw(DimensionMismatch("Iris output dimension must be two."))
    winner = 1
    y = clamp.(point, problem.branch_lower[:, 1], problem.branch_upper[:, 1])
    objective = 0.5 * sum(abs2, y .- point)
    for branch in 2:3
        candidate = clamp.(
            point,
            problem.branch_lower[:, branch],
            problem.branch_upper[:, branch],
        )
        candidate_objective = 0.5 * sum(abs2, candidate .- point)
        if candidate_objective < objective - 1e-12
            winner = branch
            y = candidate
            objective = candidate_objective
        end
    end
    return (y = collect(y), branch = winner, objective = objective)
end

function initialize_network(seed; hidden = 8)
    hidden > 0 || throw(ArgumentError("hidden width must be positive."))
    rng = MersenneTwister(820000 + seed)
    return TinyMLP(
        0.35 .* randn(rng, hidden, 2),
        zeros(hidden),
        0.25 .* randn(rng, 2, hidden),
        zeros(2),
    )
end

function _forward(network::TinyMLP, input)
    hidden = tanh.(network.W1 * input .+ network.b1)
    output = network.W2 * hidden .+ network.b2
    return output, hidden
end

function _zero_gradients(network)
    return (
        W1 = zeros(size(network.W1)),
        b1 = zeros(size(network.b1)),
        W2 = zeros(size(network.W2)),
        b2 = zeros(size(network.b2)),
    )
end

function _backward!(gradients, network, input, hidden, output_gradient)
    gradients.W2 .+= output_gradient * hidden'
    gradients.b2 .+= output_gradient
    hidden_gradient = (network.W2' * output_gradient) .* (1.0 .- hidden .^ 2)
    gradients.W1 .+= hidden_gradient * input'
    gradients.b1 .+= hidden_gradient
    return gradients
end

function _adam_state(network)
    return AdamState(
        zeros(size(network.W1)), zeros(size(network.b1)),
        zeros(size(network.W2)), zeros(size(network.b2)),
        zeros(size(network.W1)), zeros(size(network.b1)),
        zeros(size(network.W2)), zeros(size(network.b2)), 0,
    )
end

function _adam_array!(parameter, gradient, first, second, step, learning_rate, beta1, beta2)
    first .= beta1 .* first .+ (1.0 - beta1) .* gradient
    second .= beta2 .* second .+ (1.0 - beta2) .* (gradient .^ 2)
    first_hat = first ./ (1.0 - beta1^step)
    second_hat = second ./ (1.0 - beta2^step)
    parameter .-= learning_rate .* first_hat ./ (sqrt.(second_hat) .+ 1e-8)
    return nothing
end

function _adam_update!(network, gradients, state, learning_rate)
    state.step += 1
    beta1, beta2 = 0.9, 0.999
    _adam_array!(network.W1, gradients.W1, state.first_W1, state.second_W1,
        state.step, learning_rate, beta1, beta2)
    _adam_array!(network.b1, gradients.b1, state.first_b1, state.second_b1,
        state.step, learning_rate, beta1, beta2)
    _adam_array!(network.W2, gradients.W2, state.first_W2, state.second_W2,
        state.step, learning_rate, beta1, beta2)
    _adam_array!(network.b2, gradients.b2, state.first_b2, state.second_b2,
        state.step, learning_rate, beta1, beta2)
    return nothing
end

_loss(y, target) = sum(abs2, y .- target)
_loss_gradient(y, target) = 2.0 .* (y .- target)

function _record_center!(counters, center, model)
    counters.center_global_searches += 1
    counters.center_leaf_solves += center.leaf_solves
    counters.center_relaxation_solves += center.relaxation_solves
    counters.logical_failures += center.report === nothing ||
        !center.report.feasible || !check_feasibility(model, center.y).feasible
    return nothing
end

function _sample_gradient(
    method,
    problem,
    point,
    target,
    samples,
    radius,
    counters,
)
    center = lazy_branch_search(problem.model, point; store_prune_records = false)
    center.status == :optimal || error("Exact center projection was unresolved.")
    _record_center!(counters, center, problem.model)

    if method == :hard_local
        branch = project_branch(problem.model, point, center.choices)
        local_result = fixed_branch_loss_gradient(
            problem.model,
            point,
            center.choices,
            y -> _loss_gradient(y, target);
            branch_result = branch,
        )
        counters.reference_branch_solves += 1
        local_result.status == :ok || error("Hard-local gradient was unresolved.")
        return local_result.gradient, _loss(center.y, target)
    elseif method == :ste
        return _loss_gradient(center.y, target), _loss(center.y, target)
    elseif method == :blackbox
        center_loss = _loss(center.y, target)
        total = zeros(2)
        for column in axes(samples, 2)
            u = samples[:, column]
            perturbed = lazy_branch_search(
                problem.model,
                point .+ radius .* u;
                store_prune_records = false,
            )
            perturbed.status == :optimal || error("Black-box search was unresolved.")
            counters.perturbed_global_searches += 1
            counters.lazy_global_searches += 1
            counters.lazy_leaf_solves += perturbed.leaf_solves
            counters.lazy_relaxation_solves += perturbed.relaxation_solves
            counters.switching_samples += perturbed.choices != center.choices
            counters.logical_failures += perturbed.report === nothing ||
                !perturbed.report.feasible
            residual = _loss(perturbed.y, target) - center_loss
            total .+= (residual / radius) .* ball_smoothing_score(u; beta = 2.0)
        end
        return total ./ size(samples, 2), center_loss
    elseif method == :sage
        estimate = sage_gradient(
            problem.model,
            point,
            radius,
            y -> _loss(y, target),
            y -> _loss_gradient(y, target);
            samples = samples,
            center_result = center,
            store_records = false,
        )
        estimate.status == :ok || error("SAGE gradient was unresolved.")
        counters.perturbed_global_searches += estimate.perturbed_global_searches
        counters.reference_branch_solves += estimate.reference_branch_solves
        counters.competitor_branch_solves += estimate.competitor_branch_solves
        counters.sample_stability_certificates += estimate.sample_stability_certificates
        counters.restricted_global_searches += estimate.restricted_global_searches
        counters.lazy_global_searches += estimate.lazy_global_searches
        counters.lazy_leaf_solves += estimate.lazy_leaf_solves
        counters.lazy_relaxation_solves += estimate.lazy_relaxation_solves
        counters.switching_samples += estimate.switching_samples
        counters.unresolved_samples += estimate.unresolved_samples
        return estimate.gradient, _loss(center.y, target)
    end
    throw(ArgumentError("Unknown training method $(method)."))
end

function _training_batches(indices, seed, steps, batch_size)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
    rng = MersenneTwister(830000 + seed)
    return [rand(rng, indices, batch_size) for _ in 1:steps]
end

function evaluate_network(network, problem, indices; backend = true)
    isempty(indices) && throw(ArgumentError("Evaluation split cannot be empty."))
    squared_error = 0.0
    correct_branch = 0
    feasible = 0
    for index in indices
        point, _ = _forward(network, problem.inputs[:, index])
        if backend
            projection = lazy_branch_search(
                problem.model, point; store_prune_records = false,
            )
            projection.status == :optimal || error("Evaluation search was unresolved.")
            y = projection.y
            branch = only(projection.choices)
            feasible += projection.report !== nothing && projection.report.feasible
        else
            projection = analytic_project(problem, point)
            y = projection.y
            branch = projection.branch
            feasible += 1
        end
        raw_prediction = y .* problem.target_scale .+ problem.target_mean
        squared_error += sum(abs2, raw_prediction .- problem.raw_targets[:, index])
        correct_branch += branch == problem.labels[index]
    end
    return (
        mse = squared_error / (2 * length(indices)),
        branch_accuracy = correct_branch / length(indices),
        feasibility_rate = feasible / length(indices),
    )
end

function train_method(
    problem::IrisProblem,
    method::Symbol,
    seed::Integer;
    steps::Int = 60,
    batch_size::Int = 12,
    samples::Int = 8,
    radius::Real = 0.5,
    learning_rate::Real = 0.02,
    hidden::Int = 8,
    evaluation_interval::Int = 10,
)
    method in (:hard_local, :ste, :blackbox, :sage) ||
        throw(ArgumentError("Unsupported training method $(method)."))
    steps > 0 || throw(ArgumentError("steps must be positive."))
    samples > 0 && iseven(samples) ||
        throw(ArgumentError("samples must be a positive even integer."))
    radius > 0 || throw(ArgumentError("radius must be positive."))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive."))

    network = initialize_network(seed; hidden = hidden)
    optimizer = _adam_state(network)
    batches = _training_batches(problem.train, seed, steps, batch_size)
    counters = TrainingCounters()
    trajectory = NamedTuple[]

    function record_trajectory!(step)
        train_metrics = evaluate_network(network, problem, problem.train; backend = false)
        validation_metrics = evaluate_network(
            network, problem, problem.validation; backend = false,
        )
        push!(trajectory, (
            step = step,
            train_mse = train_metrics.mse,
            train_branch_accuracy = train_metrics.branch_accuracy,
            validation_mse = validation_metrics.mse,
            validation_branch_accuracy = validation_metrics.branch_accuracy,
        ))
    end

    record_trajectory!(0)
    started = time_ns()
    for step in 1:steps
        gradients = _zero_gradients(network)
        for index in batches[step]
            input = problem.inputs[:, index]
            target = problem.targets[:, index]
            point, hidden_values = _forward(network, input)
            perturbations = method in (:blackbox, :sage) ?
                sample_antithetic_ball_smoothing(
                    MersenneTwister(840000000 + 100000seed + 1000step + index),
                    2,
                    samples;
                    beta = 2.0,
                ) : zeros(2, samples)
            output_gradient, _ = _sample_gradient(
                method,
                problem,
                point,
                target,
                perturbations,
                Float64(radius),
                counters,
            )
            _backward!(gradients, network, input, hidden_values, output_gradient)
        end
        scale = 1.0 / batch_size
        gradients.W1 .*= scale
        gradients.b1 .*= scale
        gradients.W2 .*= scale
        gradients.b2 .*= scale
        _adam_update!(network, gradients, optimizer, Float64(learning_rate))
        if step % evaluation_interval == 0 || step == steps
            record_trajectory!(step)
        end
    end
    training_seconds = (time_ns() - started) / 1e9

    return (
        network = network,
        trajectory = trajectory,
        training_seconds = training_seconds,
        center_global_searches = counters.center_global_searches,
        center_leaf_solves = counters.center_leaf_solves,
        center_relaxation_solves = counters.center_relaxation_solves,
        perturbed_global_searches = counters.perturbed_global_searches,
        reference_branch_solves = counters.reference_branch_solves,
        competitor_branch_solves = counters.competitor_branch_solves,
        sample_stability_certificates = counters.sample_stability_certificates,
        restricted_global_searches = counters.restricted_global_searches,
        lazy_global_searches = counters.lazy_global_searches,
        lazy_leaf_solves = counters.lazy_leaf_solves,
        lazy_relaxation_solves = counters.lazy_relaxation_solves,
        switching_samples = counters.switching_samples,
        unresolved_samples = counters.unresolved_samples,
        logical_failures = counters.logical_failures,
    )
end

end # module
