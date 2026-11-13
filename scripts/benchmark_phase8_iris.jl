using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "phase8_iris_common.jl"))
using .Phase8Iris

function parse_seeds()
    seeds = parse.(Int, strip.(split(get(ENV, "SAGE_PHASE8_SEEDS", "1,2,3,4,5"), ',')))
    length(seeds) >= 5 || error("Phase 8 requires at least five seeds.")
    length(unique(seeds)) == length(seeds) || error("Phase 8 seeds must be unique.")
    return seeds
end

function csv_value(value)
    if value isa AbstractString || value isa Symbol
        escaped = replace(string(value), '"' => "\"\"")
        return occursin(r"[,\"\n]", escaped) ? "\"$(escaped)\"" : escaped
    end
    return string(value)
end

function write_csv(path, columns, rows)
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
end

function json_escape(value)
    return replace(
        string(value), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n",
        '\r' => "\\r", '\t' => "\\t",
    )
end

function json_value(value)
    value === nothing && return "null"
    value isa Bool && return value ? "true" : "false"
    value isa Number && return isfinite(value) ? string(value) : "null"
    value isa AbstractString && return "\"$(json_escape(value))\""
    value isa Symbol && return json_value(string(value))
    value isa AbstractVector && return "[" * join(json_value.(value), ',') * "]"
    if value isa AbstractDict
        entries = [json_value(string(key)) * ":" * json_value(item) for (key, item) in value]
        return "{" * join(entries, ',') * "}"
    end
    error("Unsupported JSON value $(typeof(value)).")
end

function paired_interval(differences)
    n = length(differences)
    n >= 5 || error("Paired confidence interval requires at least five seeds.")
    center = mean(differences)
    # Conservative t critical value for n=5; using it for larger n errs wide.
    halfwidth = 2.776 * std(differences; corrected = true) / sqrt(n)
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function method_summary(rows, method)
    selected = filter(row -> row.method == string(method), rows)
    return (
        test_mse = mean(row.test_mse for row in selected),
        test_accuracy = mean(row.test_branch_accuracy for row in selected),
        validation_mse = mean(row.validation_mse for row in selected),
        seconds = mean(row.training_seconds for row in selected),
        global_searches = sum(row.perturbed_global_searches for row in selected),
        sample_certificates = sum(row.sample_stability_certificates for row in selected),
        reference_solves = sum(row.reference_branch_solves for row in selected),
        competitor_solves = sum(row.competitor_branch_solves for row in selected),
    )
end

function row_for(rows, method, seed)
    return only(filter(
        row -> row.method == string(method) && row.seed == seed,
        rows,
    ))
end

function main()
    seeds = parse_seeds()
    steps = parse(Int, get(ENV, "SAGE_PHASE8_STEPS", "60"))
    batch_size = parse(Int, get(ENV, "SAGE_PHASE8_BATCH_SIZE", "12"))
    samples = parse(Int, get(ENV, "SAGE_PHASE8_SAMPLES", "8"))
    radius = parse(Float64, get(ENV, "SAGE_PHASE8_RADIUS", "0.5"))
    learning_rate = parse(Float64, get(ENV, "SAGE_PHASE8_LEARNING_RATE", "0.02"))
    hidden = parse(Int, get(ENV, "SAGE_PHASE8_HIDDEN", "8"))
    evaluation_interval = parse(Int, get(ENV, "SAGE_PHASE8_EVAL_INTERVAL", "10"))
    output_directory = abspath(get(
        ENV, "SAGE_PHASE8_OUTPUT", "phase8_iris_results",
    ))
    data_path = abspath(get(
        ENV,
        "SAGE_PHASE8_DATA",
        joinpath(@__DIR__, "..", "data", "phase8", "iris.data"),
    ))
    steps > 0 || error("SAGE_PHASE8_STEPS must be positive.")
    batch_size > 0 || error("SAGE_PHASE8_BATCH_SIZE must be positive.")
    samples > 0 && iseven(samples) || error("SAGE_PHASE8_SAMPLES must be positive and even.")
    radius > 0 || error("SAGE_PHASE8_RADIUS must be positive.")
    learning_rate > 0 || error("SAGE_PHASE8_LEARNING_RATE must be positive.")
    evaluation_interval > 0 || error("SAGE_PHASE8_EVAL_INTERVAL must be positive.")
    mkpath(output_directory)

    dataset = load_iris(data_path)
    methods = (:hard_local, :ste, :blackbox, :sage)
    run_rows = NamedTuple[]
    trajectory_rows = NamedTuple[]
    maximum_projection_parity_error = 0.0
    maximum_projection_objective_error = 0.0

    # Compile every gradient path before recorded training time.
    warm_problem = prepare_problem(dataset, first(seeds))
    for method in methods
        train_method(
            warm_problem,
            method,
            100 + first(seeds);
            steps = 1,
            batch_size = 2,
            samples = 2,
            radius = radius,
            learning_rate = learning_rate,
            hidden = hidden,
            evaluation_interval = 1,
        )
    end

    for seed in seeds
        problem = prepare_problem(dataset, seed)

        # Analytic/backend parity is audited on every held-out observation and
        # on the initial network predictions used by all compared methods.
        initial_network = initialize_network(seed; hidden = hidden)
        for index in vcat(problem.validation, problem.test)
            input = problem.inputs[:, index]
            hidden_values = tanh.(initial_network.W1 * input .+ initial_network.b1)
            point = initial_network.W2 * hidden_values .+ initial_network.b2
            analytic = analytic_project(problem, point)
            backend = lazy_branch_search(
                problem.model, point; store_prune_records = false,
            )
            backend.status == :optimal || error("Phase 8 parity search was unresolved.")
            only(backend.choices) == analytic.branch || error("Phase 8 branch parity failed.")
            maximum_projection_parity_error = max(
                maximum_projection_parity_error,
                norm(backend.y .- analytic.y),
            )
            maximum_projection_objective_error = max(
                maximum_projection_objective_error,
                abs(backend.objective - analytic.objective),
            )
        end

        # Rotate execution order across seeds to reduce systematic timing bias.
        shift = mod(seed - first(seeds), length(methods))
        ordered_methods = circshift(collect(methods), -shift)
        for method in ordered_methods
            initial_test = evaluate_network(
                initialize_network(seed; hidden = hidden),
                problem,
                problem.test;
                backend = true,
            )
            result = train_method(
                problem,
                method,
                seed;
                steps = steps,
                batch_size = batch_size,
                samples = samples,
                radius = radius,
                learning_rate = learning_rate,
                hidden = hidden,
                evaluation_interval = evaluation_interval,
            )
            train_metrics = evaluate_network(
                result.network, problem, problem.train; backend = true,
            )
            validation_metrics = evaluate_network(
                result.network, problem, problem.validation; backend = true,
            )
            test_metrics = evaluate_network(
                result.network, problem, problem.test; backend = true,
            )
            push!(run_rows, (
                seed = seed,
                method = string(method),
                initial_test_mse = initial_test.mse,
                train_mse = train_metrics.mse,
                validation_mse = validation_metrics.mse,
                test_mse = test_metrics.mse,
                train_branch_accuracy = train_metrics.branch_accuracy,
                validation_branch_accuracy = validation_metrics.branch_accuracy,
                test_branch_accuracy = test_metrics.branch_accuracy,
                train_feasibility = train_metrics.feasibility_rate,
                validation_feasibility = validation_metrics.feasibility_rate,
                test_feasibility = test_metrics.feasibility_rate,
                training_seconds = result.training_seconds,
                center_global_searches = result.center_global_searches,
                center_leaf_solves = result.center_leaf_solves,
                center_relaxation_solves = result.center_relaxation_solves,
                perturbed_global_searches = result.perturbed_global_searches,
                reference_branch_solves = result.reference_branch_solves,
                competitor_branch_solves = result.competitor_branch_solves,
                sample_stability_certificates = result.sample_stability_certificates,
                restricted_global_searches = result.restricted_global_searches,
                lazy_global_searches = result.lazy_global_searches,
                lazy_leaf_solves = result.lazy_leaf_solves,
                lazy_relaxation_solves = result.lazy_relaxation_solves,
                switching_samples = result.switching_samples,
                unresolved_samples = result.unresolved_samples,
                logical_failures = result.logical_failures,
            ))
            for row in result.trajectory
                push!(trajectory_rows, (
                    seed = seed,
                    method = string(method),
                    step = row.step,
                    train_mse = row.train_mse,
                    train_branch_accuracy = row.train_branch_accuracy,
                    validation_mse = row.validation_mse,
                    validation_branch_accuracy = row.validation_branch_accuracy,
                ))
            end
        end
    end

    run_path = joinpath(output_directory, "phase8_runs.csv")
    trajectory_path = joinpath(output_directory, "phase8_trajectories.csv")
    summary_path = joinpath(output_directory, "phase8_summary.json")
    run_columns = (
        :seed, :method, :initial_test_mse, :train_mse, :validation_mse, :test_mse,
        :train_branch_accuracy, :validation_branch_accuracy, :test_branch_accuracy,
        :train_feasibility, :validation_feasibility, :test_feasibility,
        :training_seconds, :center_global_searches, :center_leaf_solves,
        :center_relaxation_solves, :perturbed_global_searches,
        :reference_branch_solves, :competitor_branch_solves,
        :sample_stability_certificates,
        :restricted_global_searches, :lazy_global_searches, :lazy_leaf_solves,
        :lazy_relaxation_solves, :switching_samples, :unresolved_samples,
        :logical_failures,
    )
    trajectory_columns = (
        :seed, :method, :step, :train_mse, :train_branch_accuracy,
        :validation_mse, :validation_branch_accuracy,
    )
    write_csv(run_path, run_columns, run_rows)
    write_csv(trajectory_path, trajectory_columns, trajectory_rows)

    summaries = Dict(method => method_summary(run_rows, method) for method in methods)
    sage = summaries[:sage]
    blackbox = summaries[:blackbox]
    hard = summaries[:hard_local]
    ste = summaries[:ste]

    sage_vs_blackbox_mse = [
        row_for(run_rows, :sage, seed).test_mse -
        row_for(run_rows, :blackbox, seed).test_mse
        for seed in seeds
    ]
    sage_vs_hard_mse = [
        row_for(run_rows, :sage, seed).test_mse -
        row_for(run_rows, :hard_local, seed).test_mse
        for seed in seeds
    ]
    sage_vs_blackbox_accuracy = [
        row_for(run_rows, :sage, seed).test_branch_accuracy -
        row_for(run_rows, :blackbox, seed).test_branch_accuracy
        for seed in seeds
    ]
    mse_interval = paired_interval(sage_vs_blackbox_mse)
    hard_interval = paired_interval(sage_vs_hard_mse)
    accuracy_interval = paired_interval(sage_vs_blackbox_accuracy)

    parity_gate = maximum_projection_parity_error < 1e-6 &&
        maximum_projection_objective_error < 1e-6
    feasibility_gate = all(
        row.train_feasibility == 1.0 && row.validation_feasibility == 1.0 &&
        row.test_feasibility == 1.0 && row.logical_failures == 0 &&
        row.unresolved_samples == 0
        for row in run_rows
    )
    learning_gate = mean(
        row.test_mse / row.initial_test_mse
        for row in run_rows if row.method == "sage"
    ) <= 0.80
    matched_quality_gate = sage.test_mse <= 1.10 * blackbox.test_mse &&
        sage.test_accuracy >= blackbox.test_accuracy - 0.10
    hard_local_gate = sage.test_mse <= hard.test_mse
    blackbox_global_searches = blackbox.global_searches
    sage_global_searches = sage.global_searches
    global_search_reduction = 1.0 - sage_global_searches / blackbox_global_searches
    global_reduction_gate = global_search_reduction >= 0.25
    wall_time_ratio = sage.seconds / blackbox.seconds
    wall_time_gate = wall_time_ratio <= 1.50
    phase8_pass = parity_gate && feasibility_gate && learning_gate &&
        matched_quality_gate && hard_local_gate && global_reduction_gate &&
        wall_time_gate

    summary = Dict{String, Any}(
        "phase" => "phase8_iris_real_data_pilot",
        "julia_version" => string(VERSION),
        "dataset" => "UCI Iris",
        "dataset_doi" => "10.24432/C56C76",
        "seeds" => seeds,
        "steps" => steps,
        "batch_size" => batch_size,
        "samples" => samples,
        "radius" => radius,
        "learning_rate" => learning_rate,
        "hidden" => hidden,
        "maximum_projection_parity_error" => maximum_projection_parity_error,
        "maximum_projection_objective_error" => maximum_projection_objective_error,
        "sage_test_mse" => sage.test_mse,
        "blackbox_test_mse" => blackbox.test_mse,
        "hard_local_test_mse" => hard.test_mse,
        "ste_test_mse" => ste.test_mse,
        "sage_test_branch_accuracy" => sage.test_accuracy,
        "blackbox_test_branch_accuracy" => blackbox.test_accuracy,
        "hard_local_test_branch_accuracy" => hard.test_accuracy,
        "ste_test_branch_accuracy" => ste.test_accuracy,
        "sage_vs_blackbox_mse_difference" => mse_interval.mean,
        "sage_vs_blackbox_mse_ci95" => [mse_interval.lower, mse_interval.upper],
        "sage_vs_hard_mse_difference" => hard_interval.mean,
        "sage_vs_hard_mse_ci95" => [hard_interval.lower, hard_interval.upper],
        "sage_vs_blackbox_accuracy_difference" => accuracy_interval.mean,
        "sage_vs_blackbox_accuracy_ci95" => [accuracy_interval.lower, accuracy_interval.upper],
        "blackbox_perturbed_global_searches" => blackbox_global_searches,
        "sage_perturbed_global_searches" => sage_global_searches,
        "sage_sample_stability_certificates" => sage.sample_certificates,
        "sage_reference_branch_solves" => sage.reference_solves,
        "sage_competitor_branch_solves" => sage.competitor_solves,
        "global_search_reduction" => global_search_reduction,
        "blackbox_mean_training_seconds" => blackbox.seconds,
        "sage_mean_training_seconds" => sage.seconds,
        "wall_time_ratio" => wall_time_ratio,
        "parity_gate" => parity_gate,
        "feasibility_gate" => feasibility_gate,
        "learning_gate" => learning_gate,
        "matched_quality_gate" => matched_quality_gate,
        "hard_local_gate" => hard_local_gate,
        "global_reduction_gate" => global_reduction_gate,
        "wall_time_gate" => wall_time_gate,
        "pass" => phase8_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 8 UCI Iris real-data pilot")
    println("Julia version: ", VERSION)
    println("Dataset: UCI Iris, DOI 10.24432/C56C76")
    println("Task: sepal -> petal regression with train-only species-box disjunctions")
    println("Proposal contract: exact feasible forward, compact q beta=2, fixed rho, Eq.(17)-(18)")
    println("seeds/steps/batch/samples/rho/lr/hidden: ",
        length(seeds), "/", steps, "/", batch_size, "/", samples, "/",
        radius, "/", learning_rate, "/", hidden)
    println()
    println("method\ttest MSE\ttest branch accuracy\ttraining seconds\tperturbed global searches\tsample certificates")
    for method in methods
        item = summaries[method]
        println("$(method)\t$(item.test_mse)\t$(item.test_accuracy)\t$(item.seconds)\t$(item.global_searches)\t$(item.sample_certificates)")
    end
    println()
    println("SAGE-black-box paired test-MSE difference/95% CI: ",
        mse_interval.mean, " / [", mse_interval.lower, ", ", mse_interval.upper, "]")
    println("SAGE-hard-local paired test-MSE difference/95% CI: ",
        hard_interval.mean, " / [", hard_interval.lower, ", ", hard_interval.upper, "]")
    println("SAGE-black-box paired branch-accuracy difference/95% CI: ",
        accuracy_interval.mean, " / [", accuracy_interval.lower, ", ", accuracy_interval.upper, "]")
    println("maximum projection y/objective parity error: ",
        maximum_projection_parity_error, "/", maximum_projection_objective_error)
    println("black-box/SAGE perturbed global searches: ",
        blackbox_global_searches, "/", sage_global_searches)
    println("global-search reduction: ", global_search_reduction)
    println("SAGE/black-box mean training-time ratio: ", wall_time_ratio)
    println("parity gate: ", parity_gate ? "PASS" : "FAIL")
    println("feasibility/resolution gate: ", feasibility_gate ? "PASS" : "FAIL")
    println("SAGE learning-from-initialization gate: ", learning_gate ? "PASS" : "FAIL")
    println("matched black-box quality gate: ", matched_quality_gate ? "PASS" : "FAIL")
    println("hard-local improvement gate: ", hard_local_gate ? "PASS" : "FAIL")
    println("global-search reduction gate: ", global_reduction_gate ? "PASS" : "FAIL")
    println("training wall-time gate: ", wall_time_gate ? "PASS" : "FAIL")
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("summary JSON: ", summary_path)
    println("PHASE8_IRIS_PILOT=", phase8_pass ? "PASS" : "FAIL")
end

main()
