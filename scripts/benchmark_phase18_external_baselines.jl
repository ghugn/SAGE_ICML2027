using DisjunctiveNet
using Serialization
using Statistics
using SHA
using LinearAlgebra

include(joinpath(@__DIR__, "phase10_jobshop_common.jl"))
include(joinpath(@__DIR__, "phase18_external_common.jl"))
using .Phase10JobShop
import .Phase18External as EB

const PHASE18_SCHEMA = 2
const PHASE18_LABELS = (:hard_local, :ste, :compact_blackbox, :dbb_projection,
    :dpo_gaussian, :sage_n4)
const PHASE18_CONFIRMATORY_SEEDS = collect(11:15)
const PHASE18_DEVELOPMENT_SEEDS = [1, 2]
const PHASE18_TUNING_STEPS = 128
const PHASE18_DEADLINE = Ref(Inf)

function profile_specs(label)
    base = EB.default_spec(label)
    candidates = EB.candidate_specs(label)
    # Central configuration plus largest scale and both LR endpoints. These
    # discovery-only probes bound planning costs better than one default run.
    scale_extreme = label == :dbb_projection ? merge(base, (lambda = 5.0,)) :
        label == :dpo_gaussian ? merge(base, (gaussian_scale = 2.0,)) :
        label in (:sage_n4, :compact_blackbox) ? merge(base, (radius = 1.0,)) : base
    return unique([base, scale_extreme, first(candidates), last(candidates)])
end

function repository_revision()
    try
        return readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function source_fingerprint()
    root = normpath(joinpath(@__DIR__, ".."))
    files = [joinpath(root, name) for name in ("Project.toml", "Manifest.toml")]
    for directory in ("src", "scripts")
        for (folder, _, names) in walkdir(joinpath(root, directory))
            append!(files, [joinpath(folder, name) for name in names if endswith(name, ".jl")])
        end
    end
    buffer = IOBuffer()
    for path in sort(files)
        write(buffer, replace(relpath(path, root), '\\' => '/'), '\0', read(path), '\0')
    end
    return bytes2hex(sha256(take!(buffer)))
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

function json(value)
    value === nothing && return "null"
    value isa Bool && return value ? "true" : "false"
    value isa Number && return isfinite(value) ? string(value) : "null"
    if value isa AbstractString || value isa Symbol
        escaped = replace(string(value), '\\' => "\\\\", '"' => "\\\"",
            '\n' => "\\n", '\r' => "\\r", '\t' => "\\t")
        return "\"" * escaped * "\""
    end
    value isa AbstractVector && return "[" * join(json.(value), ',') * "]"
    if value isa NamedTuple || value isa AbstractDict
        entries = sort(collect(pairs(value)); by = pair -> string(first(pair)))
        return "{" * join([json(string(key)) * ":" * json(item)
            for (key, item) in entries], ',') * "}"
    end
    error("Unsupported JSON type $(typeof(value)).")
end

function write_json(path, value)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        println(io, json(value))
    end
    mv(temporary, path; force = true)
end

function write_csv(path, rows)
    isempty(rows) && return
    temporary = path * ".tmp"
    columns = propertynames(first(rows))
    escape(value) = "\"" * replace(string(value), '"' => "\"\"") * "\""
    open(temporary, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join([escape(getproperty(row, column)) for column in columns], ','))
        end
    end
    mv(temporary, path; force = true)
end

function process_cpu_seconds()
    if Sys.iswindows()
        # MSVC clock() measures wall time, so it cannot measure CPU use.
        handle = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
        created, exited, kernel, user = (Ref{UInt64}(0) for _ in 1:4)
        ok = ccall((:GetProcessTimes, "kernel32"), Int32,
            (Ptr{Cvoid}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}),
            handle, created, exited, kernel, user)
        ok != 0 || error("GetProcessTimes failed.")
        return (kernel[] + user[]) / 1e7
    end
    return ccall(:clock, Clong, ()) / 1_000_000
end

function resource_snapshot()
    memory = if Sys.islinux() && isfile("/proc/self/status")
        text = read("/proc/self/status", String)
        matched = match(r"VmRSS:\s+(\d+)", text)
        matched === nothing ? 0 : parse(Int, matched.captures[1]) * 1024
    else
        # Julia GC accounting is reported separately; this is not RSS.
        0
    end
    return (cpu_seconds = process_cpu_seconds(),
        rss_bytes = memory, julia_live_bytes = Base.gc_live_bytes(),
        threads = Threads.nthreads())
end

function safety(result, evaluation)
    evaluation.csat == 1.0 && evaluation.unresolved == 0 &&
        evaluation.maximum_violation <= 1e-6 && result.logical_failures == 0 &&
        result.unresolved_samples == 0 && isfinite(evaluation.task_loss)
end

function smoke(directory)
    problem, pretrained = EB.setup_problem(1; small = true)
    rows = NamedTuple[]
    for label in PHASE18_LABELS
        spec = EB.default_spec(label)
        states = Any[]
        result = EB.train_external(problem, pretrained, spec, 1; steps = 4,
            state_checkpoint_interval = 2,
            state_checkpoint_callback = state -> push!(states, state))
        resumed = EB.train_external(problem, pretrained, spec, 1; steps = 4,
            resume_state = states[1])
        evaluation = evaluate_network(result.network, problem.iid; config = problem.config)
        parity = EB.network_equal(result.network, resumed.network) &&
            result.complete_branch_solves == resumed.complete_branch_solves
        parity && safety(result, evaluation) || error("Phase 18 smoke failed for $(label).")
        path = joinpath(directory, "smoke_$(label).jls")
        save_atomic(path, (network = result.network, state = states[end]))
        EB.network_equal(open(deserialize, path).network, result.network) ||
            error("Checkpoint reload mismatch.")
        push!(rows, (method = string(label), steps = result.steps,
            validation_loss = evaluation.task_loss, csat = evaluation.csat,
            seconds = result.training_seconds, searches = result.perturbed_global_searches,
            observed_threads = result.sample_threads_used, resume_parity = parity))
        println("[phase18-smoke] ", last(rows)); flush(stdout)
    end
    write_csv(joinpath(directory, "phase18_smoke.csv"), rows)
    write_json(joinpath(directory, "phase18_smoke_summary.json"),
        (status = "PASS", methods = length(rows), source = source_fingerprint()))
    println("PHASE18_EXTERNAL_SMOKE=PASS")
end

function preflight(directory)
    starting_source = source_fingerprint()
    Threads.nthreads() >= 2 || error("Start Julia with --threads=auto (at least 2 threads).")
    steps = parse(Int, get(ENV, "SAGE_PHASE18_PREFLIGHT_STEPS", "16"))
    10 <= steps <= 50 || error("Preflight must use 10-50 representative updates.")
    problem, pretrained = EB.setup_problem(1)
    rates = Dict{String, Float64}()
    observed = Dict{String, Int}()
    work = Dict{String, Any}()
    probe_rates = Dict{String, Any}()
    parity = true
    speedup = NaN
    starting_resources = resource_snapshot()
    started = time()
    for label in PHASE18_LABELS
        spec = EB.default_spec(label)
        EB.train_external(problem, pretrained, spec, 1; steps = 1) # compilation warmup
        result = EB.train_external(problem, pretrained, spec, 1; steps = steps)
        rates[string(label)] = result.training_seconds / steps
        observed[string(label)] = result.sample_threads_used
        work[string(label)] = (center_searches = result.center_global_searches,
            perturbed_searches = result.perturbed_global_searches,
            branch_solves = result.complete_branch_solves,
            relaxation_solves = result.center_relaxation_solves + result.lazy_relaxation_solves)
        if label in (:sage_n4, :dpo_gaussian, :compact_blackbox)
            serial = EB.train_external(problem, pretrained, spec, 1;
                steps = steps, parallel = false)
            parity &= EB.network_equal(serial.network, result.network) &&
                serial.complete_branch_solves == result.complete_branch_solves
            label == :sage_n4 && (speedup = serial.training_seconds / result.training_seconds)
        end
        state_path = joinpath(directory, "preflight_$(label)_state.jls")
        EB.train_external(problem, pretrained, spec, 1; steps = steps ÷ 2,
            state_checkpoint_interval = steps ÷ 2,
            state_checkpoint_callback = state -> save_atomic(state_path, state))
        restored = EB.train_external(problem, pretrained, spec, 1; steps = steps,
            resume_state = open(deserialize, state_path))
        parity &= EB.network_equal(result.network, restored.network) &&
            result.complete_branch_solves == restored.complete_branch_solves
        probes = [(spec = spec, seconds_per_update = rates[string(label)])]
        for alternative in profile_specs(label)
            alternative == spec && continue
            probe = EB.train_external(problem, pretrained, alternative, 1; steps = steps)
            push!(probes, (spec = alternative,
                seconds_per_update = probe.training_seconds / steps))
        end
        probe_rates[string(label)] = probes
        rates[string(label)] = maximum(probe.seconds_per_update for probe in probes)
        println("[phase18-preflight] method=$(label) s/update=$(round(rates[string(label)]; digits=4)) " *
            "observed_threads=$(observed[string(label)]) parity=$(parity)"); flush(stdout)
    end
    # Conservative 2x allowance: past Phase 17's linear ETA underestimated the
    # complete run. Include equal development tuning and three real-time replays.
    tuning_seconds = length(PHASE18_DEVELOPMENT_SEEDS) * PHASE18_TUNING_STEPS *
        sum(length(EB.candidate_specs(label)) * rates[string(label)] for label in PHASE18_LABELS)
    confirmatory_seconds = 5 * 1024 * (sum(values(rates)) + 3rates["sage_n4"])
    pilot_seconds = 256 * sum(values(rates))
    predicted_seconds = 2.0 * (tuning_seconds + pilot_seconds + confirmatory_seconds)
    max_hours = parse(Float64, get(ENV, "SAGE_PHASE18_MAX_HOURS", "4.0"))
    eta_gate = predicted_seconds <= max_hours * 3600
    thread_gate = all(observed[string(label)] >= 2
        for label in (:sage_n4, :dpo_gaussian, :compact_blackbox))
    ending_resources = resource_snapshot()
    cpu_utilization = 100 * (ending_resources.cpu_seconds - starting_resources.cpu_seconds) /
        max(time() - started, eps())
    source_fingerprint() == starting_source || error("Source changed during preflight; repeat it.")
    plan = (schema = PHASE18_SCHEMA, source = starting_source, git_commit = repository_revision(),
        julia = string(VERSION), threads = Threads.nthreads(), preflight_steps = steps,
        rates = rates, observed_threads = observed, solver_work = work,
        profiled_configurations = probe_rates, tuning_steps = PHASE18_TUNING_STEPS,
        tuning_candidates_per_method = 9,
        serial_parallel_resume_parity = parity,
        sage_speedup = speedup, predicted_hours = predicted_seconds / 3600,
        max_hours = max_hours, eta_gate = eta_gate, thread_gate = thread_gate,
        approximate_cpu_percent = cpu_utilization, resources = ending_resources,
        passed = parity && thread_gate && eta_gate)
    save_atomic(joinpath(directory, "phase18_preflight_plan.jls"), plan)
    write_json(joinpath(directory, "phase18_preflight_plan.json"), plan)
    println("SAGE end-to-end serial/threaded speedup: $(speedup)")
    println("Predicted tuning + confirmatory hours (2x margin): $(plan.predicted_hours)")
    println("CPU decision: independent CPU-bound QP searches; GPU path absent.")
    println("PHASE18_PREFLIGHT=", plan.passed ? "PASS" : "FAIL")
    plan.passed || error("Preflight failed: inspect parity, actual threads and ETA before a long run.")
end

function checked_plan(directory)
    path = joinpath(directory, "phase18_preflight_plan.jls")
    isfile(path) || error("Run Phase 18 preflight first.")
    plan = open(deserialize, path)
    plan.source == source_fingerprint() || error("Source changed since preflight; rerun preflight.")
    plan.julia == string(VERSION) && plan.threads == Threads.nthreads() ||
        error("Julia version/thread configuration changed since preflight.")
    plan.passed || error("Preflight did not pass.")
    return plan
end

function execute_method(directory, problem, pretrained, spec, seed;
        steps, tag, wall_seconds = Inf)
    signature = (schema = PHASE18_SCHEMA, source = source_fingerprint(),
        julia = string(VERSION), threads = Threads.nthreads(), seed = seed,
        spec = spec, steps = steps, tag = tag, wall_seconds = wall_seconds)
    stem = "seed_$(seed)_$(spec.label)_$(tag)"
    state_path = joinpath(directory, "training_states", stem * ".jls")
    model_path = joinpath(directory, "model_checkpoints", stem * ".jls")
    cache_path = joinpath(directory, "completed_methods", stem * ".jls")
    if isfile(cache_path)
        cached = open(deserialize, cache_path)
        cached.signature == signature || error("Completed cache signature mismatch: $(stem)")
        model = open(deserialize, model_path)
        model.signature == signature || error("Model signature mismatch: $(stem)")
        println("[phase18-resume] completed $(stem)"); flush(stdout)
        return cached.row
    end
    resume_state = if isfile(state_path)
        payload = open(deserialize, state_path)
        payload.signature == signature || error("Resume signature mismatch: $(stem)")
        println("[phase18-resume] $(stem) step=$(payload.state.completed_steps)")
        payload.state
    else
        nothing
    end
    resource_start = resource_snapshot()
    wall_started = time()
    callback = state -> begin
        save_atomic(state_path, (signature = signature, state = state))
        resources = resource_snapshot()
        progress_seconds = state.elapsed_seconds
        projected = isfinite(wall_seconds) ? max(0.0, wall_seconds - progress_seconds) :
            progress_seconds / max(state.completed_steps, 1) * (steps - state.completed_steps)
        cpu_percent = 100 * (resources.cpu_seconds - resource_start.cpu_seconds) /
            max(time() - wall_started, eps())
        println("[phase18-progress] $(stem) step=$(state.completed_steps)/$(steps) " *
            "elapsed=$(round(progress_seconds;digits=1)) ETA_s=$(round(projected;digits=1)) " *
            "CPU%=$(round(cpu_percent;digits=1)) RSS_MB=$(round(resources.rss_bytes/2.0^20;digits=1)) " *
            "checkpoint=saved threads=$(state.counters.sample_threads_used)")
        flush(stdout)
        time() <= PHASE18_DEADLINE[] || error(
            "Phase 18 session deadline reached; the current training state was saved. Resume unchanged.")
    end
    result = EB.train_external(problem, pretrained, spec, seed; steps = steps,
        wall_time_limit_seconds = wall_seconds, resume_state = resume_state,
        state_checkpoint_interval = 16, state_checkpoint_callback = callback)
    validation_only = startswith(tag, "tune") || tag == "pilot"
    evaluation_started = time_ns()
    iid = evaluate_network(result.network, problem.iid; config = problem.config)
    ood = validation_only ? iid :
        evaluate_network(result.network, problem.ood; config = problem.config)
    safety(result, iid) && safety(result, ood) || error("Safety failure: $(stem)")
    row = (seed = seed, method = string(spec.label), tag = tag,
        steps = result.steps, learning_rate = spec.learning_rate, samples = spec.samples,
        radius = spec.radius, lambda = spec.lambda, gaussian_scale = spec.gaussian_scale,
        iid_loss = iid.task_loss,
        ood_loss = validation_only ? NaN : ood.task_loss,
        ood_makespan_gap = validation_only ? NaN : ood.makespan_gap,
        csat = min(iid.csat, ood.csat), maximum_violation = max(iid.maximum_violation, ood.maximum_violation),
        seconds = result.training_seconds, budget_seconds = wall_seconds,
        evaluation_seconds = (time_ns() - evaluation_started) / 1e9,
        center_searches = result.center_global_searches,
        perturbed_searches = result.perturbed_global_searches,
        branch_solves = result.complete_branch_solves,
        relaxation_solves = result.center_relaxation_solves + result.lazy_relaxation_solves,
        observed_threads = result.sample_threads_used, stop_reason = string(result.stop_reason))
    save_atomic(model_path, (signature = signature, network = result.network, row = row))
    reloaded = open(deserialize, model_path)
    EB.network_equal(result.network, reloaded.network) || error("Model reload mismatch.")
    reproduced = evaluate_network(reloaded.network, problem.iid; config = problem.config)
    reproduced.task_loss == iid.task_loss || error("Reloaded model metric mismatch.")
    save_atomic(cache_path, (signature = signature, row = row))
    println("[phase18] $(stem) steps=$(row.steps) seconds=$(round(row.seconds;digits=2)) " *
        "IID=$(row.iid_loss) OOD=$(row.ood_loss)"); flush(stdout)
    return row
end

function select_candidates(rows; labels = PHASE18_LABELS, seeds = PHASE18_DEVELOPMENT_SEEDS,
        steps = PHASE18_TUNING_STEPS)
    selected = Dict{Symbol, Any}()
    scores = Dict{String, Any}()
    for label in labels
        candidates = EB.candidate_specs(label)
        means = Float64[]
        for (index, spec) in enumerate(candidates)
            losses = Float64[]
            for seed in seeds
                matches = filter(row -> row.seed == seed && row.method == string(label) &&
                    row.tag == "tune$(index)", rows)
                length(matches) == 1 || error("Missing or duplicate validation row: $(label)/$(index)/$(seed)")
                row = only(matches)
                row.steps == steps && row.learning_rate == spec.learning_rate &&
                    row.radius == spec.radius && row.lambda == spec.lambda &&
                    row.gaussian_scale == spec.gaussian_scale && row.samples == spec.samples ||
                    error("Validation configuration mismatch: $(label)/$(index)/$(seed)")
                row.csat == 1.0 && isfinite(row.iid_loss) && isnan(row.ood_loss) ||
                    error("Validation selection requires finite feasible validation-only rows.")
                push!(losses, row.iid_loss)
            end
            push!(means, mean(losses))
        end
        winner = argmin(means)
        selected[label] = candidates[winner]
        scores[string(label)] = (candidate_validation_losses = means,
            candidates = candidates, winner = winner, selected = selected[label])
    end
    return selected, scores
end

function tune(directory)
    plan = checked_plan(directory)
    isfinite(PHASE18_DEADLINE[]) || (PHASE18_DEADLINE[] = time() + plan.max_hours * 3600)
    selection_path = joinpath(directory, "phase18_selection.jls")
    if isfile(selection_path)
        frozen = open(deserialize, selection_path)
        frozen.source == plan.source || error("Frozen selection source mismatch.")
        println("[phase18-resume] frozen validation selection loaded")
        return frozen.selected
    end
    # A repeated tune uses exact completed-method caches, never a partially
    # selected set. Neither confirmatory IID nor OOD is constructed here.
    rows = NamedTuple[]
    for seed in PHASE18_DEVELOPMENT_SEEDS
        problem, pretrained = EB.setup_problem(seed)
        for label in PHASE18_LABELS, (index, spec) in enumerate(EB.candidate_specs(label))
            push!(rows, execute_method(directory, problem, pretrained, spec, seed;
                steps = PHASE18_TUNING_STEPS, tag = "tune$(index)"))
            write_csv(joinpath(directory, "phase18_validation.csv"), rows)
        end
    end
    selected, scores = select_candidates(rows)
    save_atomic(selection_path, (source = plan.source, selected = selected))
    write_json(joinpath(directory, "phase18_selection.json"),
        (schema = PHASE18_SCHEMA, development_seeds = PHASE18_DEVELOPMENT_SEEDS,
            validation_steps = PHASE18_TUNING_STEPS,
            validation_pool_seed = 18_020_002, candidates_per_method = 9,
            total_validation_training_seconds = sum(row.seconds for row in rows),
            total_validation_perturbed_searches = sum(row.perturbed_searches for row in rows),
            source = plan.source, scores = scores))
    println("PHASE18_VALIDATION_SELECTION=COMPLETE")
    return selected
end

function interval(values)
    length(values) == 5 || error("Registered paired intervals require exactly five seeds.")
    center = mean(values)
    halfwidth = 2.776445105 * std(values; corrected = true) / sqrt(5)
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function report(directory, rows)
    complete = length(rows) == 5 * (length(PHASE18_LABELS) + 3)
    comparisons = Dict{String, Any}()
    secondary_comparisons = Dict{String, Any}()
    if complete
        row_for(seed, label, tag) = only(filter(row -> row.seed == seed &&
            row.method == string(label) && row.tag == tag, rows))
        # Two registered literature-family comparisons, Bonferroni familywise
        # correction via paired 97.5% two-sided t intervals (df=4).
        for label in (:dbb_projection, :dpo_gaussian)
            values = [row_for(seed, :sage_n4, "fixed").ood_loss -
                row_for(seed, label, "fixed").ood_loss for seed in PHASE18_CONFIRMATORY_SEEDS]
            raw = interval(values)
            halfwidth = 3.495405933 * std(values; corrected = true) / sqrt(5)
            comparisons[string(label)] = (difference = raw.mean,
                paired_ci95 = [raw.lower, raw.upper],
                familywise_ci = [raw.mean - halfwidth, raw.mean + halfwidth],
                superiority_gate = raw.mean + halfwidth < 0)
        end
        for label in (:ste, :dbb_projection, :dpo_gaussian)
            values = [row_for(seed, :sage_n4, "fixed").ood_loss -
                row_for(seed, label, "matched_time").ood_loss for seed in PHASE18_CONFIRMATORY_SEEDS]
            secondary_comparisons[string(label)] = interval(values)
        end
    end
    aggregates = Dict{String, Any}()
    for label in PHASE18_LABELS, tag in ("fixed", "matched_time")
        selected = filter(row -> row.method == string(label) && row.tag == tag, rows)
        isempty(selected) && continue
        aggregates["$(label)/$(tag)"] = (runs = length(selected),
            mean_iid_loss = mean(row.iid_loss for row in selected),
            mean_ood_loss = mean(row.ood_loss for row in selected),
            mean_seconds = mean(row.seconds for row in selected),
            total_perturbed_searches = sum(row.perturbed_searches for row in selected),
            total_branch_solves = sum(row.branch_solves for row in selected))
    end
    matched = filter(row -> row.tag == "matched_time", rows)
    matched_time_gate = !isempty(matched) && all(row ->
        row.stop_reason == "wall_time" && row.seconds <= 1.10row.budget_seconds, matched)
    summary = (phase = "phase18_external_baselines", schema = PHASE18_SCHEMA,
        source = source_fingerprint(), git_commit = repository_revision(),
        status = complete ? "COMPLETE" : "PARTIAL",
        seeds = PHASE18_CONFIRMATORY_SEEDS, fixed_steps = 1024,
        safety_gate = all(row -> row.csat == 1.0 && row.maximum_violation <= 1e-6, rows),
        matched_time_overshoot_gate = matched_time_gate, comparisons = comparisons,
        matched_time_exploratory_ci95 = secondary_comparisons, aggregates = aggregates,
        baseline_scope = "DBB quadratic-projection adaptation; DPO-inspired Gaussian hard-forward VJP",
        external_dataset = false)
    write_csv(joinpath(directory, "phase18_runs.csv"), rows)
    write_json(joinpath(directory, "phase18_summary.json"), summary)
    return summary
end

function remaining_hours(rows)
    labels = sort(unique(row.method for row in rows))
    labels == sort(string.(collect(PHASE18_LABELS))) ||
        error("Selected-settings pilot must cover all six methods exactly once.")
    length(rows) == length(PHASE18_LABELS) || error("Duplicate selected-settings pilot rows.")
    all(row -> row.steps == 256 && row.tag == "pilot" && row.csat == 1.0 &&
        isfinite(row.iid_loss) && isnan(row.ood_loss) && row.seconds > 0, rows) ||
        error("Invalid validation-only selected-settings pilot.")
    rates = Dict(row.method => row.seconds / row.steps for row in rows)
    return 2.0 * 5 * 1024 * (sum(values(rates)) + 3rates["sage_n4"]) / 3600
end

function pilot(directory, selected)
    plan = checked_plan(directory)
    problem, pretrained = EB.setup_problem(1)
    rows = NamedTuple[]
    for label in PHASE18_LABELS
        EB.train_external(problem, pretrained, selected[label], 1; steps = 1)
        push!(rows, execute_method(directory, problem, pretrained, selected[label], 1;
            steps = 256, tag = "pilot"))
    end
    write_csv(joinpath(directory, "phase18_pilot.csv"), rows)
    hours = remaining_hours(rows)
    record = (schema = PHASE18_SCHEMA, source = plan.source, selected = selected,
        predicted_confirmatory_hours = hours, maximum_hours = plan.max_hours,
        passed = hours <= plan.max_hours, rows = rows)
    save_atomic(joinpath(directory, "phase18_pilot_plan.jls"), record)
    write_json(joinpath(directory, "phase18_pilot_plan.json"), record)
    println("Selected-settings pilot confirmatory ETA (2x margin): $(hours) hours")
    println("PHASE18_EXTERNAL_PILOT=COMPLETE")
    record.passed || error("Selected-settings pilot exceeds the time ceiling; held-out execution is locked.")
    return record
end

function confirmatory(directory)
    lowercase(get(ENV, "SAGE_PHASE18_ALLOW_CONFIRMATORY", "0")) in ("1", "true") ||
        error("Set SAGE_PHASE18_ALLOW_CONFIRMATORY=1 after reviewing preflight and validation plan.")
    plan = checked_plan(directory)
    PHASE18_DEADLINE[] = time() + plan.max_hours * 3600
    selected = tune(directory)
    # Selected configurations can have different long-horizon solver costs
    # than 16-update default probes. Never open held-out pools before this gate.
    pilot(directory, selected)
    rows = NamedTuple[]
    function check_session_budget()
        time() < PHASE18_DEADLINE[] || error(
            "Session time budget reached; completed methods and training states are saved. Resume unchanged.")
    end
    for seed in PHASE18_CONFIRMATORY_SEEDS
        check_session_budget()
        problem, pretrained = EB.setup_problem(seed; pool = :heldout)
        # Warm every method outside all reported training budgets.
        for label in PHASE18_LABELS
            EB.train_external(problem, pretrained, selected[label], seed; steps = 1)
        end
        for label in circshift(collect(PHASE18_LABELS), -mod(seed, length(PHASE18_LABELS)))
            check_session_budget()
            push!(rows, execute_method(directory, problem, pretrained, selected[label], seed;
                steps = 1024, tag = "fixed"))
            report(directory, rows)
        end
        sage = only(filter(row -> row.seed == seed && row.method == "sage_n4", rows))
        for label in circshift([:ste, :dbb_projection, :dpo_gaussian], -mod(seed, 3))
            check_session_budget()
            push!(rows, execute_method(directory, problem, pretrained, selected[label], seed;
                steps = 16384, tag = "matched_time", wall_seconds = sage.seconds))
            report(directory, rows)
        end
    end
    summary = report(directory, rows)
    println("method/budget\tmean OOD loss\tmean seconds\tperturbed searches")
    for (name, aggregate) in sort(collect(summary.aggregates); by = first)
        println("$(name)\t$(aggregate.mean_ood_loss)\t$(aggregate.mean_seconds)\t" *
            "$(aggregate.total_perturbed_searches)")
    end
    println(json(summary))
    println("PHASE18_EXTERNAL_CONFIRMATORY=COMPLETE")
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE18_PROFILE", "smoke")))
    directory = abspath(get(ENV, "SAGE_PHASE18_OUTPUT",
        joinpath(@__DIR__, "..", "artifacts", "local_runs", "phase18_external_$(profile)")))
    mkpath(directory)
    # Avoid nested BLAS pools around independent solver/sample work.
    BLAS.set_num_threads(1)
    profile == :smoke && return smoke(directory)
    profile == :preflight && return preflight(directory)
    profile == :tune && return tune(directory)
    if profile == :pilot
        selected = tune(directory)
        return pilot(directory, selected)
    end
    profile == :confirmatory && return confirmatory(directory)
    error("SAGE_PHASE18_PROFILE must be smoke, preflight, tune, pilot, or confirmatory.")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
