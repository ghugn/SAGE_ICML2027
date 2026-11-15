using DisjunctiveNet
using Serialization
using Statistics

include(joinpath(@__DIR__, "phase10_jobshop_common.jl"))
using .Phase10JobShop

const PHASE17_SCHEMA = 1
const SAMPLE_LABELS = (:sage_n2, :sage_n4, :sage_n8, :sage_n16, :sage_n32)

function repository_revision()
    repository = normpath(joinpath(@__DIR__, ".."))
    try
        return readchomp(`git -C $repository rev-parse HEAD`)
    catch
        return "unknown"
    end
end

csv_value(value) = value isa AbstractString || value isa Symbol ?
    (occursin(r"[,\"\n]", string(value)) ?
        "\"$(replace(string(value), '\"' => "\"\""))\"" : string(value)) : string(value)

function write_csv(path, rows)
    isempty(rows) && return
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
end

function atomic_serialize(path, value)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        serialize(io, value)
        flush(io)
    end
    mv(temporary, path; force = true)
    return path
end

json_escape(value) = replace(
    string(value), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n",
    '\r' => "\\r", '\t' => "\\t",
)

function json_value(value)
    value === nothing && return "null"
    value isa Bool && return value ? "true" : "false"
    value isa Number && return isfinite(value) ? string(value) : "null"
    value isa AbstractString && return "\"$(json_escape(value))\""
    value isa Symbol && return json_value(string(value))
    value isa AbstractVector && return "[" * join(json_value.(value), ',') * "]"
    if value isa AbstractDict
        entries = [json_value(string(key)) * ":" * json_value(item)
                   for (key, item) in sort!(collect(value); by = pair -> string(first(pair)))]
        return "{" * join(entries, ',') * "}"
    end
    error("Unsupported JSON value $(typeof(value)).")
end

function write_json(path, value)
    open(path, "w") do io
        println(io, json_value(value))
    end
end

network_equal(left::JobShopMLP, right::JobShopMLP) = all(
    getproperty(left, name) == getproperty(right, name)
    for name in (:W1, :b1, :W2, :b2, :W3, :b3)
)

method_spec(label) = label == :ste_updates ? (backend = :ste, samples = 2) :
    label == :ste_time ? (backend = :ste, samples = 2) :
    label == :sage_n2 ? (backend = :sage_multiscale, samples = 2) :
    label == :sage_n4 ? (backend = :sage_multiscale, samples = 4) :
    label == :sage_n8 ? (backend = :sage_multiscale, samples = 8) :
    label == :sage_n16 ? (backend = :sage_multiscale, samples = 16) :
    label == :sage_n32 ? (backend = :sage_multiscale, samples = 32) :
    throw(ArgumentError("Unknown Phase 17 method $(label)."))

function setup_problem(seed; train_size = 16, iid_size = 12, ood_size = 12,
        pretrain_epochs = 8)
    config = JobShopConfig(n_jobs = 3, n_machines = 4)
    problem = prepare_problem(
        seed; config = config, train_size = train_size,
        iid_size = iid_size, ood_size = ood_size,
    )
    pretrained = pretrain_network(
        problem, seed; epochs = pretrain_epochs,
        batch_size = min(8, train_size), learning_rate = 2e-3,
    )
    return problem, pretrained
end

function warm_paths(problem, pretrained; radius = 0.5, parallel = false)
    train_method(
        problem, pretrained, :ste, 17_000;
        max_steps = 1, samples = 2, radius = radius, evaluation_interval = 1,
    )
    train_method(
        problem, pretrained, :sage_multiscale, 17_000;
        max_steps = 1, samples = 4, radius = radius, evaluation_interval = 1,
        parallel_sage_samples = parallel,
    )
end

function paired_interval(values)
    center = mean(values)
    length(values) >= 5 || return (mean = center, lower = NaN, upper = NaN)
    halfwidth = 2.776 * std(values; corrected = true) / sqrt(length(values))
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function smoke(output_directory)
    problem, pretrained = setup_problem(
        1; train_size = 4, iid_size = 4, ood_size = 4, pretrain_epochs = 1,
    )
    warm_paths(problem, pretrained)
    rows = NamedTuple[]
    for label in (:ste_updates, :sage_n2, :sage_n4)
        spec = method_spec(label)
        result = train_method(
            problem, pretrained, spec.backend, 1;
            epochs = 1, max_steps = 4, samples = spec.samples,
            radius = 0.5, learning_rate = 1e-3,
            evaluation_interval = 2, checkpoint_steps = [4],
        )
        evaluation = evaluate_network(
            result.network, problem.ood; config = problem.config,
        )
        push!(rows, (
            method = string(label), samples = spec.samples, steps = result.steps,
            ood_loss = evaluation.task_loss, ood_csat = evaluation.csat,
            seconds = result.training_seconds,
            global_searches = result.perturbed_global_searches,
            logical_failures = result.logical_failures,
        ))
    end
    write_csv(joinpath(output_directory, "phase17_smoke.csv"), rows)
    passed = length(rows) == 3 && all(
        row.ood_csat == 1.0 && row.logical_failures == 0 && isfinite(row.ood_loss)
        for row in rows
    )
    println("SAGE Phase 17 sample-efficiency smoke")
    println("methods/rows: 3/", length(rows))
    println("safety gate: ", passed ? "PASS" : "FAIL")
    println("PHASE17_SAMPLE_EFFICIENCY_SMOKE=", passed ? "PASS" : "FAIL")
    passed || exit(1)
end

function preflight(output_directory)
    Threads.nthreads() >= 2 || error(
        "Phase 17 preflight requires at least two Julia threads. Start Julia with --threads=auto.",
    )
    # Seed 1 belongs to the Phase 16 discovery set. Confirmatory seeds 6-10
    # remain untouched by system calibration and ETA planning.
    calibration_seed = 1
    problem, pretrained = setup_problem(calibration_seed)
    warm_paths(problem, pretrained; parallel = true)
    steps = parse(Int, get(ENV, "SAGE_PHASE17_PREFLIGHT_STEPS", "64"))
    steps >= 8 || throw(ArgumentError("Preflight needs at least 8 steps."))
    half = fld(steps, 2)

    serial = train_method(
        problem, pretrained, :sage_multiscale, calibration_seed;
        epochs = cld(steps, 16), max_steps = steps, samples = 4,
        radius = 0.5, learning_rate = 1e-3,
        evaluation_interval = steps, checkpoint_steps = [steps],
        parallel_sage_samples = false,
    )
    threaded = train_method(
        problem, pretrained, :sage_multiscale, calibration_seed;
        epochs = cld(steps, 16), max_steps = steps, samples = 4,
        radius = 0.5, learning_rate = 1e-3,
        evaluation_interval = steps, checkpoint_steps = [steps],
        parallel_sage_samples = true,
    )
    parity = network_equal(serial.network, threaded.network) &&
        serial.perturbed_global_searches == threaded.perturbed_global_searches &&
        serial.complete_branch_solves == threaded.complete_branch_solves
    speedup = serial.training_seconds / threaded.training_seconds

    resume_path = joinpath(output_directory, "phase17_resume_probe.jls")
    train_method(
        problem, pretrained, :sage_multiscale, calibration_seed;
        epochs = cld(half, 16), max_steps = half, samples = 4,
        radius = 0.5, learning_rate = 1e-3,
        evaluation_interval = half, checkpoint_steps = [half],
        parallel_sage_samples = true,
        state_checkpoint_interval = half,
        state_checkpoint_callback = state -> begin
            atomic_serialize(resume_path, (
                schema = PHASE17_SCHEMA, seed = calibration_seed,
                samples = 4, state = state,
            ))
        end,
    )
    stored = open(deserialize, resume_path)
    stored.schema == PHASE17_SCHEMA || error("Phase 17 resume probe schema mismatch.")
    resumed = train_method(
        problem, pretrained, :sage_multiscale, calibration_seed;
        epochs = cld(steps, 16), max_steps = steps, samples = 4,
        radius = 0.5, learning_rate = 1e-3,
        evaluation_interval = steps, checkpoint_steps = [half, steps],
        parallel_sage_samples = true, resume_state = stored.state,
    )
    resume_parity = network_equal(threaded.network, resumed.network) &&
        threaded.perturbed_global_searches == resumed.perturbed_global_searches &&
        threaded.complete_branch_solves == resumed.complete_branch_solves

    timing = Dict{String, Any}()
    ste = train_method(
        problem, pretrained, :ste, calibration_seed;
        epochs = cld(steps, 16), max_steps = steps, samples = 2,
        radius = 0.5, learning_rate = 1e-3, evaluation_interval = steps,
    )
    timing["ste"] = ste.training_seconds / steps
    for samples in (2, 4, 8, 16, 32)
        result = samples == 4 ? threaded : train_method(
            problem, pretrained, :sage_multiscale, calibration_seed;
            epochs = cld(steps, 16), max_steps = steps, samples = samples,
            radius = 0.5, learning_rate = 1e-3, evaluation_interval = steps,
            parallel_sage_samples = true,
        )
        timing["sage_n$(samples)"] = result.training_seconds / steps
    end

    matched_time_steps = clamp(
        round(Int, 1024 * timing["sage_n4"] / timing["ste"]), 1024, 8192,
    )
    # Include a conservative 25% systems margin for state serialization,
    # evaluation, pretraining and shared-machine timing drift.
    eta_safety_factor = 1.25
    predicted_seconds = eta_safety_factor * 5 * (
        sum(1024 * timing["sage_n$(samples)"] for samples in (2, 4, 8, 16, 32)) +
        1024 * timing["ste"] + matched_time_steps * timing["ste"]
    )
    maximum_eta_hours = parse(Float64, get(ENV, "SAGE_PHASE17_MAX_ETA_HOURS", "4.0"))
    parity_gate = parity && resume_parity
    speed_gate = speedup >= 1.3
    thread_gate = threaded.sample_threads_used >= 2
    eta_gate = predicted_seconds <= maximum_eta_hours * 3600
    passed = parity_gate && speed_gate && thread_gate && eta_gate

    plan = (
        schema = PHASE17_SCHEMA,
        git_commit = repository_revision(),
        calibration_seed = calibration_seed,
        julia_version = string(VERSION),
        threads = Threads.nthreads(),
        preflight_steps = steps,
        serial_seconds = serial.training_seconds,
        threaded_seconds = threaded.training_seconds,
        speedup = speedup,
        parity_gate = parity_gate,
        resume_parity_gate = resume_parity,
        thread_gate = thread_gate,
        speed_gate = speed_gate,
        eta_gate = eta_gate,
        maximum_eta_hours = maximum_eta_hours,
        predicted_seconds = predicted_seconds,
        eta_safety_factor = eta_safety_factor,
        matched_time_ste_steps = matched_time_steps,
        per_step_seconds = timing,
        passed = passed,
    )
    plan_path = atomic_serialize(
        joinpath(output_directory, "phase17_preflight_plan.jls"), plan,
    )
    write_json(joinpath(output_directory, "phase17_preflight_plan.json"), Dict(
        "schema" => plan.schema, "git_commit" => plan.git_commit,
        "calibration_seed" => plan.calibration_seed,
        "julia_version" => plan.julia_version,
        "threads" => plan.threads, "preflight_steps" => plan.preflight_steps,
        "serial_seconds" => plan.serial_seconds,
        "threaded_seconds" => plan.threaded_seconds, "speedup" => plan.speedup,
        "parity_gate" => plan.parity_gate,
        "resume_parity_gate" => plan.resume_parity_gate,
        "thread_gate" => plan.thread_gate, "speed_gate" => plan.speed_gate,
        "eta_gate" => plan.eta_gate,
        "maximum_eta_hours" => plan.maximum_eta_hours,
        "predicted_seconds" => plan.predicted_seconds,
        "eta_safety_factor" => plan.eta_safety_factor,
        "matched_time_ste_steps" => plan.matched_time_ste_steps,
        "per_step_seconds" => timing, "passed" => plan.passed,
    ))

    println("SAGE Phase 17 systems preflight")
    println("threads/observed sample threads: ", Threads.nthreads(), "/",
        threaded.sample_threads_used)
    println("serial/threaded seconds: ", serial.training_seconds, "/",
        threaded.training_seconds)
    println("end-to-end speedup: ", speedup)
    println("serial/threaded parity gate: ", parity ? "PASS" : "FAIL")
    println("resume parity gate: ", resume_parity ? "PASS" : "FAIL")
    println("speedup >= 1.3 gate: ", speed_gate ? "PASS" : "FAIL")
    println("predicted confirmatory hours: ", predicted_seconds / 3600)
    println("ETA <= $(maximum_eta_hours)h gate: ", eta_gate ? "PASS" : "FAIL")
    println("frozen matched-time STE steps: ", matched_time_steps)
    println("preflight plan: ", plan_path)
    println("PHASE17_PREFLIGHT=", passed ? "PASS" : "FAIL")
    passed || exit(1)
end

function run_confirmatory_method(
    output_directory, problem, pretrained, seed, label, steps;
    radius = 0.5, learning_rate = 1e-3,
)
    spec = method_spec(label)
    signature = (
        schema = PHASE17_SCHEMA, git_commit = repository_revision(),
        seed = seed, method = string(label),
        steps = steps, samples = spec.samples, radius = radius,
        learning_rate = learning_rate, threads = Threads.nthreads(),
    )
    cache_directory = joinpath(output_directory, "completed_methods")
    state_directory = joinpath(output_directory, "training_states")
    model_directory = joinpath(output_directory, "model_checkpoints")
    foreach(mkpath, (cache_directory, state_directory, model_directory))
    cache_path = joinpath(cache_directory, "seed_$(seed)_$(label).jls")
    state_path = joinpath(state_directory, "seed_$(seed)_$(label).jls")
    model_path = joinpath(model_directory, "seed_$(seed)_$(label)_final.jls")
    if isfile(cache_path)
        payload = open(deserialize, cache_path)
        payload.signature == signature || error(
            "Phase 17 completed-method signature mismatch at $(cache_path).",
        )
        isfile(model_path) || error(
            "Phase 17 completed cache exists but its model artifact is missing at $(model_path).",
        )
        model_payload = open(deserialize, model_path)
        model_payload.signature == signature || error(
            "Phase 17 model signature mismatch at $(model_path).",
        )
        cached_evaluation = evaluate_network(
            model_payload.network, problem.ood; config = problem.config,
        )
        cached_evaluation.task_loss == payload.row.ood_loss || error(
            "Phase 17 cached model metric mismatch at $(model_path).",
        )
        println("[phase17-resume] seed=$(seed) method=$(label) loaded completed cache")
        return payload.row
    end
    resume_state = nothing
    if isfile(state_path)
        payload = open(deserialize, state_path)
        payload.signature == signature || error(
            "Phase 17 training-state signature mismatch at $(state_path).",
        )
        resume_state = payload.state
        println("[phase17-resume] seed=$(seed) method=$(label) step=$(resume_state.completed_steps)")
    end
    checkpoint_interval = min(64, steps)
    callback = state -> begin
        atomic_serialize(state_path, (signature = signature, state = state))
        println(
            "[phase17-progress] seed=$(seed) method=$(label) " *
            "step=$(state.completed_steps)/$(steps) " *
            "seconds=$(round(state.elapsed_seconds; digits = 1))",
        )
        flush(stdout)
    end
    result = train_method(
        problem, pretrained, spec.backend, seed;
        epochs = cld(steps, size(problem.train.inputs, 2)),
        max_steps = steps, samples = spec.samples, radius = radius,
        learning_rate = learning_rate, evaluation_interval = 64,
        checkpoint_steps = [steps], parallel_sage_samples = spec.backend != :ste,
        resume_state = resume_state, state_checkpoint_interval = checkpoint_interval,
        state_checkpoint_callback = callback,
    )
    evaluation = evaluate_network(result.network, problem.ood; config = problem.config)
    row = (
        seed = seed, method = string(label), samples = spec.samples, steps = steps,
        ood_loss = evaluation.task_loss, ood_makespan_gap = evaluation.makespan_gap,
        ood_csat = evaluation.csat, maximum_violation = evaluation.maximum_violation,
        seconds = result.training_seconds,
        global_searches = result.perturbed_global_searches,
        branch_solves = result.complete_branch_solves,
        switching_samples = result.switching_samples,
        mean_ess = result.mean_effective_sample_size,
        unresolved = result.unresolved_samples + evaluation.unresolved,
        logical_failures = result.logical_failures,
        sample_threads_used = result.sample_threads_used,
    )
    atomic_serialize(model_path, (
        signature = signature, network = result.network, metrics = row,
    ))
    reloaded = open(deserialize, model_path)
    network_equal(reloaded.network, result.network) || error(
        "Phase 17 model checkpoint reload mismatch at $(model_path).",
    )
    reloaded_evaluation = evaluate_network(
        reloaded.network, problem.ood; config = problem.config,
    )
    reloaded_evaluation.task_loss == evaluation.task_loss || error(
        "Phase 17 model checkpoint metric mismatch at $(model_path).",
    )
    atomic_serialize(cache_path, (signature = signature, row = row))
    println(
        "[phase17] seed=$(seed) method=$(label) steps=$(steps) " *
        "seconds=$(round(result.training_seconds; digits = 1)) " *
        "loss=$(round(evaluation.task_loss; digits = 6))",
    )
    flush(stdout)
    return row
end

function confirmatory(output_directory)
    lowercase(get(ENV, "SAGE_PHASE17_ALLOW_CONFIRMATORY", "0")) in
        ("1", "true", "yes") || error(
        "Set SAGE_PHASE17_ALLOW_CONFIRMATORY=1 only after reviewing a passing preflight plan.",
    )
    Threads.nthreads() >= 2 || error(
        "Phase 17 confirmatory run requires Julia --threads=auto.",
    )
    plan_path = abspath(get(ENV, "SAGE_PHASE17_PREFLIGHT", ""))
    isfile(plan_path) || error("SAGE_PHASE17_PREFLIGHT must point to phase17_preflight_plan.jls.")
    plan = open(deserialize, plan_path)
    plan.schema == PHASE17_SCHEMA || error("Phase 17 preflight schema mismatch.")
    plan.git_commit == repository_revision() || error(
        "Phase 17 code revision differs from the reviewed preflight plan.",
    )
    plan.passed || error("Phase 17 confirmatory run is locked because preflight failed.")
    plan.threads == Threads.nthreads() || error(
        "Phase 17 thread count differs from the reviewed preflight plan.",
    )

    seeds = collect(6:10)
    fixed_steps = 1024
    labels = (:ste_updates, :ste_time, SAMPLE_LABELS...)
    rows = NamedTuple[]
    for seed in seeds
        problem, pretrained = setup_problem(seed)
        warm_paths(problem, pretrained; parallel = true)
        shift = mod(seed, length(labels))
        for label in circshift(collect(labels), -shift)
            steps = label == :ste_time ? plan.matched_time_ste_steps : fixed_steps
            push!(rows, run_confirmatory_method(
                output_directory, problem, pretrained, seed, label, steps,
            ))
            write_csv(joinpath(output_directory, "phase17_runs.csv"), rows)
        end
    end

    row_for(label, seed) = only(filter(
        row -> row.method == string(label) && row.seed == seed, rows,
    ))
    n4_minus_ste = paired_interval([
        row_for(:sage_n4, seed).ood_loss - row_for(:ste_updates, seed).ood_loss
        for seed in seeds
    ])
    n4_minus_ste_time = paired_interval([
        row_for(:sage_n4, seed).ood_loss - row_for(:ste_time, seed).ood_loss
        for seed in seeds
    ])
    n4_minus_n32 = paired_interval([
        row_for(:sage_n4, seed).ood_loss - row_for(:sage_n32, seed).ood_loss
        for seed in seeds
    ])
    noninferiority_margin = 0.005
    safety_gate = all(
        row.ood_csat == 1.0 && row.maximum_violation <= 1e-6 &&
        row.unresolved == 0 && row.logical_failures == 0
        for row in rows
    )
    checkpoint_gate = all(
        isfile(joinpath(
            output_directory, "model_checkpoints",
            "seed_$(seed)_$(label)_final.jls",
        )) for seed in seeds, label in labels
    )
    primary_superiority = n4_minus_ste.upper < 0
    matched_time_superiority = n4_minus_ste_time.upper < 0
    n4_noninferior_n32 = n4_minus_n32.upper < noninferiority_margin
    n4_searches = sum(row_for(:sage_n4, seed).global_searches for seed in seeds)
    n32_searches = sum(row_for(:sage_n32, seed).global_searches for seed in seeds)
    search_reduction = 1 - n4_searches / n32_searches
    sample_efficiency_gate = n4_noninferior_n32 && search_reduction >= 0.5

    summary = Dict{String, Any}(
        "phase" => "phase17_sample_efficiency",
        "status" => "COMPLETE",
        "seeds" => seeds,
        "fixed_steps" => fixed_steps,
        "matched_time_ste_steps" => plan.matched_time_ste_steps,
        "primary_n4_minus_ste_ci95" => [
            n4_minus_ste.mean, n4_minus_ste.lower, n4_minus_ste.upper,
        ],
        "matched_time_n4_minus_ste_ci95" => [
            n4_minus_ste_time.mean, n4_minus_ste_time.lower, n4_minus_ste_time.upper,
        ],
        "n4_minus_n32_ci95" => [
            n4_minus_n32.mean, n4_minus_n32.lower, n4_minus_n32.upper,
        ],
        "n4_vs_n32_noninferiority_margin" => noninferiority_margin,
        "n4_vs_n32_global_search_reduction" => search_reduction,
        "primary_superiority_gate" => primary_superiority,
        "matched_time_superiority_gate" => matched_time_superiority,
        "n4_noninferiority_gate" => n4_noninferior_n32,
        "sample_efficiency_gate" => sample_efficiency_gate,
        "safety_gate" => safety_gate,
        "model_checkpoint_gate" => checkpoint_gate,
    )
    write_json(joinpath(output_directory, "phase17_summary.json"), summary)

    println("SAGE Phase 17 confirmatory sample-efficiency frontier")
    println("new seeds: ", seeds)
    println("fixed/matched-time STE steps: ", fixed_steps, "/", plan.matched_time_ste_steps)
    println("method\tmean OOD loss\tmean seconds\tglobal searches")
    for label in labels
        selected = filter(row -> row.method == string(label), rows)
        println("$(label)\t$(mean(row.ood_loss for row in selected))\t" *
            "$(mean(row.seconds for row in selected))\t" *
            "$(sum(row.global_searches for row in selected))")
    end
    println("N4 - fixed-step STE OOD/95% CI: ", n4_minus_ste.mean,
        " / [", n4_minus_ste.lower, ", ", n4_minus_ste.upper, "]")
    println("N4 - matched-time STE OOD/95% CI: ", n4_minus_ste_time.mean,
        " / [", n4_minus_ste_time.lower, ", ", n4_minus_ste_time.upper, "]")
    println("N4 - N32 OOD/95% CI: ", n4_minus_n32.mean,
        " / [", n4_minus_n32.lower, ", ", n4_minus_n32.upper, "]")
    println("N4/N32 global-search reduction: ", search_reduction)
    for (name, gate) in (
        ("safety", safety_gate), ("model checkpoint", checkpoint_gate),
        ("primary N4 superiority", primary_superiority),
        ("matched-time N4 superiority", matched_time_superiority),
        ("N4/N32 noninferiority", n4_noninferior_n32),
        ("sample efficiency", sample_efficiency_gate),
    )
        println(name, " gate: ", gate ? "PASS" : "FAIL")
    end
    println("PHASE17_CONFIRMATORY=COMPLETE")
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE17_PROFILE", "smoke")))
    output_directory = abspath(get(
        ENV, "SAGE_PHASE17_OUTPUT", "phase17_sample_efficiency_$(profile)_results",
    ))
    mkpath(output_directory)
    profile == :smoke && return smoke(output_directory)
    profile == :preflight && return preflight(output_directory)
    profile == :confirmatory && return confirmatory(output_directory)
    throw(ArgumentError(
        "SAGE_PHASE17_PROFILE must be smoke, preflight, or confirmatory.",
    ))
end

main()
