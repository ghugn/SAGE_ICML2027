using DisjunctiveNet
using LinearAlgebra
using Serialization
using SHA
using Statistics

include(joinpath(@__DIR__, "phase10_jobshop_common.jl"))
include(joinpath(@__DIR__, "phase18_external_common.jl"))
include(joinpath(@__DIR__, "phase19_ablation_common.jl"))
using .Phase10JobShop
using .Phase18External
using .Phase19Ablation

const PHASE19_SCHEMA = 2

function json(value)
    value === nothing && return "null"
    value isa Bool && return value ? "true" : "false"
    value isa Number && return isfinite(value) ? string(value) : "null"
    value isa Symbol && return json(string(value))
    if value isa AbstractString
        escaped = replace(value, '\\' => "\\\\", '"' => "\\\"",
            '\n' => "\\n", '\r' => "\\r", '\t' => "\\t")
        return "\"$(escaped)\""
    end
    if value isa NamedTuple
        return "{" * join((json(string(name)) * ":" * json(getproperty(value, name))
            for name in propertynames(value)), ",") * "}"
    end
    if value isa AbstractDict
        return "{" * join((json(string(key)) * ":" * json(item)
            for (key, item) in sort(collect(value); by = pair -> string(first(pair)))), ",") * "}"
    end
    if value isa Tuple || value isa AbstractVector
        return "[" * join((json(item) for item in value), ",") * "]"
    end
    error("Unsupported JSON value $(typeof(value)).")
end

function write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        write(io, json(value), '\n')
    end
    return path
end

csv_cell(value) = begin
    text = value isa AbstractFloat && !isfinite(value) ? "" : string(value)
    occursin(r"[,\"\n\r]", text) ? "\"" * replace(text, '"' => "\"\"") * "\"" : text
end

function write_csv(path, rows)
    isempty(rows) && error("Cannot write an empty Phase 19 CSV.")
    names = propertynames(first(rows))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(names), ','))
        for row in rows
            propertynames(row) == names || error("CSV rows have inconsistent fields.")
            println(io, join((csv_cell(getproperty(row, name)) for name in names), ','))
        end
    end
    return path
end

function save_atomic(path, value)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        serialize(io, value)
        flush(io)
    end
    mv(temporary, path; force = true)
    return path
end

function source_fingerprint()
    root = normpath(joinpath(@__DIR__, ".."))
    relative = [
        "scripts/phase10_jobshop_common.jl",
        "scripts/phase18_external_common.jl",
        "scripts/phase19_ablation_common.jl",
        "scripts/benchmark_phase19_component_ablation.jl",
        "src/gradients/sage_estimator.jl",
        "src/gradients/geometry_sampling.jl",
        "src/certificates/branch_margin.jl",
        "src/search/lazy_branch_search.jl",
    ]
    buffer = IOBuffer()
    for name in relative
        path = joinpath(root, split(name, '/')...)
        write(buffer, name, '\0', read(path), '\0')
    end
    return bytes2hex(sha256(take!(buffer)))
end

function locate_phase18_results(path)
    root = abspath(path)
    isfile(joinpath(root, "phase18_selection.jls")) && return root
    isdir(root) || error("Phase 18 bundle path does not exist: $(root)")
    matches = String[]
    for (directory, _, files) in walkdir(root)
        "phase18_selection.jls" in files && push!(matches, directory)
    end
    length(matches) == 1 || error(
        "Expected exactly one Phase 18 result directory under $(root), found $(length(matches)).",
    )
    return only(matches)
end

function checkpoint_path(results, seed, label)
    path = joinpath(results, "model_checkpoints", "seed_$(seed)_$(label)_fixed.jls")
    isfile(path) || error("Missing Phase 18 checkpoint $(path).")
    return path
end

function cache_protocol(signature)
    return (
        seed = signature.seed,
        checkpoint = signature.checkpoint,
        checkpoint_hash = signature.checkpoint_hash,
        index = signature.index,
        radius = signature.radius,
        repetitions = signature.repetitions,
        reference_samples = signature.reference_samples,
        estimator_samples = signature.estimator_samples,
    )
end

function cache_is_compatible(previous, current)
    previous == current && return true
    # Schema 2 only changes the treatment of an all-undefined cosine summary.
    # Completed schema-1 centers contain valid estimator outputs and may be
    # resumed without recomputing them when every experimental field matches.
    return previous.schema == 1 && current.schema == 2 &&
        cache_protocol(previous) == cache_protocol(current)
end

function profile_config(profile)
    if profile == "smoke"
        return (
            tasks = [(seed = 11, checkpoint = :compact_blackbox, index = 3)],
            repetitions = 2,
            reference_samples = 16,
            estimator_samples = 4,
        )
    elseif profile == "confirm"
        tasks = NamedTuple[]
        for seed in 11:15
            push!(tasks, (seed = seed, checkpoint = :compact_blackbox, index = 3))
            push!(tasks, (seed = seed, checkpoint = :sage_n4, index = 11))
        end
        return (
            tasks = tasks,
            repetitions = 16,
            reference_samples = 256,
            estimator_samples = 4,
        )
    end
    throw(ArgumentError("SAGE_PHASE19_PROFILE must be smoke or confirm."))
end

function raw_csv_row(task, row)
    return (
        seed = task.seed,
        checkpoint = task.checkpoint,
        train_index = task.index,
        variant = row.variant,
        repetition = row.repetition,
        relative_error = row.relative_error,
        cosine = row.cosine,
        gradient_norm = row.gradient_norm,
        seconds = row.seconds,
        global_searches = row.global_searches,
        unique_branch_solves = row.branch_solves,
        relaxation_solves = row.relaxation_solves,
        switching_samples = row.switching_samples,
        sample_certificates = row.sample_certificates,
        ball_certified = row.ball_certified,
        pruned_fraction = row.pruned_fraction,
        ess = row.ess,
        workers = row.workers,
    )
end

function task_csv_row(task, reference, row)
    return merge(task, (
        reference_half_disagreement = reference.half_disagreement,
        reference_seconds = reference.seconds,
        reference_global_searches = reference.global_searches,
        reference_branch_solves = reference.branch_solves,
    ), row)
end

function aggregate_rows(task_rows)
    rows = NamedTuple[]
    for variant in PHASE19_VARIANTS
        selected = filter(row -> row.variant == variant, task_rows)
        push!(rows, (
            variant = variant,
            centers = length(selected),
            mean_relative_bias = mean(row.relative_bias for row in selected),
            mean_trace_variance = mean(row.trace_variance for row in selected),
            mean_rmse = mean(row.rmse for row in selected),
            mean_cosine = Phase19Ablation._finite_mean(row.mean_cosine for row in selected),
            mean_seconds = mean(row.mean_seconds for row in selected),
            mean_global_searches = mean(row.mean_global_searches for row in selected),
            mean_unique_branch_solves = mean(row.mean_branch_solves for row in selected),
            mean_sample_certificates = mean(row.mean_sample_certificates for row in selected),
            ball_certificate_rate = mean(row.ball_certificate_rate for row in selected),
            mean_pruned_fraction = mean(row.mean_pruned_fraction for row in selected),
            mean_ess = mean(row.mean_ess for row in selected),
            maximum_workers = maximum(row.maximum_workers for row in selected),
        ))
    end
    base = only(filter(row -> row.variant == :compact_blackbox, rows))
    return [merge(row, (
        global_search_reduction_vs_compact = 1.0 -
            row.mean_global_searches / base.mean_global_searches,
        branch_solve_reduction_vs_compact = 1.0 -
            row.mean_unique_branch_solves / base.mean_unique_branch_solves,
        time_ratio_vs_compact = row.mean_seconds / base.mean_seconds,
    )) for row in rows]
end

function run_task(task, config, results, output, radius, source)
    checkpoint = checkpoint_path(results, task.seed, task.checkpoint)
    checkpoint_hash = bytes2hex(sha256(read(checkpoint)))
    signature = (
        schema = PHASE19_SCHEMA,
        source = source,
        seed = task.seed,
        checkpoint = task.checkpoint,
        checkpoint_hash = checkpoint_hash,
        index = task.index,
        radius = radius,
        repetitions = config.repetitions,
        reference_samples = config.reference_samples,
        estimator_samples = config.estimator_samples,
    )
    cache = joinpath(
        output,
        "completed_centers",
        "seed_$(task.seed)_$(task.checkpoint)_index_$(task.index).jls",
    )
    if isfile(cache)
        payload = deserialize(cache)
        cache_is_compatible(payload.signature, signature) ||
            error("Phase 19 center cache signature mismatch: $(cache)")
        payload.signature == signature || println(
            "[phase19-resume] accepted schema-1 center after metric-only NaN fix: $(cache)",
        )
        println("[phase19-resume] seed=$(task.seed) checkpoint=$(task.checkpoint) index=$(task.index)")
        return payload.result
    end

    problem, _ = setup_problem(task.seed; pool = :heldout, small = false)
    payload = deserialize(checkpoint)
    network = payload.network
    context = center_context(problem, network, task.index)
    started = time_ns()
    result = ablate_task(
        context,
        1_000task.seed + task.index;
        radius = radius,
        estimator_samples = config.estimator_samples,
        repetitions = config.repetitions,
        reference_samples = config.reference_samples,
        parallel = true,
        progress = (variant, repetition, repetitions, estimate) -> begin
            if repetition == 1 || repetition == repetitions || repetition % 4 == 0
                println(
                    "[phase19-progress] seed=$(task.seed) checkpoint=$(task.checkpoint) " *
                    "index=$(task.index) variant=$(variant) repetition=$(repetition)/$(repetitions) " *
                    "searches=$(estimate.global_searches) branches=$(estimate.branch_solves) " *
                    "seconds=$(round(estimate.seconds; digits = 2))",
                )
                flush(stdout)
            end
        end,
    )
    elapsed = (time_ns() - started) / 1e9
    stored = (task = task, reference = result.reference, raw = result.raw,
        summaries = result.summaries, elapsed = elapsed)
    save_atomic(cache, (signature = signature, result = stored))
    println(
        "[phase19] completed seed=$(task.seed) checkpoint=$(task.checkpoint) " *
        "index=$(task.index) seconds=$(round(elapsed; digits = 1)) checkpoint=saved",
    )
    flush(stdout)
    return stored
end

function main()
    profile = get(ENV, "SAGE_PHASE19_PROFILE", "smoke")
    config = profile_config(profile)
    output = abspath(get(ENV, "SAGE_PHASE19_OUTPUT",
        joinpath(@__DIR__, "..", "phase19_component_ablation_results")))
    bundle = get(ENV, "SAGE_PHASE18_BUNDLE",
        joinpath(@__DIR__, "..", "SAGE_Phase18_fixed_complete"))
    results = locate_phase18_results(bundle)
    selection = deserialize(joinpath(results, "phase18_selection.jls"))
    radius = Float64(selection.selected[:sage_n4].radius)
    estimator_samples = Int(selection.selected[:sage_n4].samples)
    compact = selection.selected[:compact_blackbox]
    compact.radius == radius && compact.samples == estimator_samples || error(
        "Phase 18 compact/SAGE q-rho-N settings are not matched.",
    )
    estimator_samples == config.estimator_samples || error(
        "Phase 19 must retain the frozen Phase 18 N=$(estimator_samples).",
    )
    source = source_fingerprint()
    mkpath(output)
    println("SAGE Phase 19 component ablation")
    println("profile/tasks/threads: $(profile)/$(length(config.tasks))/$(Threads.nthreads())")
    println("frozen rho/N: $(radius)/$(estimator_samples)")
    println("reference/repetitions: $(config.reference_samples)/$(config.repetitions)")
    println("Phase 18 results: $(results)")
    println("No wall-time gate and no performance pass/fail gate.")

    task_results = NamedTuple[]
    total_started = time_ns()
    for (task_index, task) in enumerate(config.tasks)
        task_started = time_ns()
        push!(task_results, run_task(task, config, results, output, radius, source))
        elapsed = (time_ns() - task_started) / 1e9
        completed = task_index
        total_elapsed = (time_ns() - total_started) / 1e9
        estimated_remaining = completed == 0 ? NaN : total_elapsed / completed *
            (length(config.tasks) - completed)
        println(
            "[phase19-task] $(completed)/$(length(config.tasks)) elapsed=$(round(elapsed; digits=1)) " *
            "informational_ETA_s=$(round(estimated_remaining; digits=1))",
        )
        flush(stdout)
    end

    raw_rows = NamedTuple[]
    task_rows = NamedTuple[]
    for result in task_results
        append!(raw_rows, [raw_csv_row(result.task, row) for row in result.raw])
        append!(task_rows, [task_csv_row(result.task, result.reference, row)
            for row in result.summaries])
    end
    aggregate = aggregate_rows(task_rows)
    references = [result.reference for result in task_results]
    safety = all(row -> row.unresolved == 0 && row.logical_failures == 0,
        Iterators.flatten(result.raw for result in task_results))
    safety || error("Phase 19 safety invariant failed.")
    summary = (
        phase = "phase19_component_ablation",
        schema = PHASE19_SCHEMA,
        status = "COMPLETE",
        profile = profile,
        source = source,
        protocol = (
            centers = length(config.tasks),
            checkpoint_sources = ["compact_blackbox", "sage_n4"],
            radius = radius,
            estimator_samples = estimator_samples,
            repetitions = config.repetitions,
            reference_samples = config.reference_samples,
            variants = collect(string.(PHASE19_VARIANTS)),
            exact_forward = true,
            common_random_numbers = true,
        ),
        reference = (
            mean_half_disagreement = mean(row.half_disagreement for row in references),
            maximum_half_disagreement = maximum(row.half_disagreement for row in references),
            total_global_searches = sum(row.global_searches for row in references),
            total_unique_branch_solves = sum(row.branch_solves for row in references),
        ),
        safety_gate = safety,
        aggregate = aggregate,
        interpretation_policy = "diagnostic_only_no_automatic_winner",
        branch_accounting = "unique QP solves; SAGE lazy leaves are not double-counted",
    )
    write_csv(joinpath(output, "phase19_raw_estimates.csv"), raw_rows)
    write_csv(joinpath(output, "phase19_center_summaries.csv"), task_rows)
    write_csv(joinpath(output, "phase19_aggregate.csv"), aggregate)
    write_json(joinpath(output, "phase19_summary.json"), summary)
    write_json(joinpath(output, "PHASE19_MANIFEST.json"), (
        status = "PHASE19_COMPONENT_ABLATION_COMPLETE",
        source = source,
        profile = profile,
        summary_sha256 = bytes2hex(sha256(read(joinpath(output, "phase19_summary.json")))),
    ))

    println("\nvariant\trel_bias\tvariance\trmse\tcosine\tsearches\tbranches\ttime_ratio")
    for row in aggregate
        println(join((
            row.variant,
            round(row.mean_relative_bias; digits = 4),
            round(row.mean_trace_variance; digits = 4),
            round(row.mean_rmse; digits = 4),
            round(row.mean_cosine; digits = 4),
            round(row.mean_global_searches; digits = 2),
            round(row.mean_unique_branch_solves; digits = 2),
            round(row.time_ratio_vs_compact; digits = 3),
        ), '\t'))
    end
    println("reference mean/max half disagreement: " *
        "$(summary.reference.mean_half_disagreement)/$(summary.reference.maximum_half_disagreement)")
    println("safety gate: PASS")
    println("results: $(output)")
    println("PHASE19_ABLATION=COMPLETE")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
