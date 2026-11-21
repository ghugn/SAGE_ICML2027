using DisjunctiveNet
using Serialization
using Statistics
using SHA
using LinearAlgebra

include(joinpath(@__DIR__, "phase10_jobshop_common.jl"))
include(joinpath(@__DIR__, "phase18_external_common.jl"))
using .Phase10JobShop
import .Phase18External as EB

const PHASE18_SCHEMA = 3
const PHASE18_LABELS = (:hard_local, :ste, :compact_blackbox, :dbb_projection,
    :dpo_gaussian, :sage_n4)
const PHASE18_CONFIRMATORY_SEEDS = collect(11:15)
const PHASE18_DEVELOPMENT_SEEDS = [1, 2]
const PHASE18_TUNING_STEPS = 128
# Exact scientific source of the imported partial bundle. No estimator, data,
# selection, training-budget, or evaluation setting changed after this source.
const PHASE18_LEGACY_CACHE_SOURCES = Set([
    "7f66c687fa82ab98270e9d44e29d78a822b997540f31be980657ccd42a946a63",
    "de6f422b4efdc91c9a46c4728979504af42af2084ec1213c67deed11661ed0d0",
])

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

cache_source_compatible(source, current = source_fingerprint()) =
    source == current || source in PHASE18_LEGACY_CACHE_SOURCES

function scientific_signature(signature)
    ignored = (:source, :threads, :wall_seconds)
    return (; (name => getproperty(signature, name)
        for name in propertynames(signature) if !(name in ignored))...)
end

function signature_compatible(saved, expected)
    hasproperty(saved, :source) && cache_source_compatible(saved.source, expected.source) ||
        return false
    return scientific_signature(saved) == scientific_signature(expected)
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

function execute_method(directory, problem, pretrained, spec, seed; steps, tag)
    signature = (schema = PHASE18_SCHEMA, source = source_fingerprint(),
        julia = string(VERSION), seed = seed, spec = spec, steps = steps, tag = tag)
    stem = "seed_$(seed)_$(spec.label)_$(tag)"
    state_path = joinpath(directory, "training_states", stem * ".jls")
    model_path = joinpath(directory, "model_checkpoints", stem * ".jls")
    cache_path = joinpath(directory, "completed_methods", stem * ".jls")
    if isfile(cache_path)
        cached = open(deserialize, cache_path)
        signature_compatible(cached.signature, signature) ||
            error("Completed cache signature mismatch: $(stem)")
        model = open(deserialize, model_path)
        signature_compatible(model.signature, signature) ||
            error("Model signature mismatch: $(stem)")
        println("[phase18-resume] completed $(stem)"); flush(stdout)
        return cached.row
    end
    resume_state = if isfile(state_path)
        payload = open(deserialize, state_path)
        signature_compatible(payload.signature, signature) ||
            error("Resume signature mismatch: $(stem)")
        println("[phase18-resume] $(stem) step=$(payload.state.completed_steps)")
        payload.state
    else
        nothing
    end
    callback = state -> begin
        save_atomic(state_path, (signature = signature, state = state))
        println("[phase18-progress] $(stem) step=$(state.completed_steps)/$(steps) " *
            "elapsed=$(round(state.elapsed_seconds;digits=1)) checkpoint=saved")
        flush(stdout)
    end
    result = EB.train_external(problem, pretrained, spec, seed; steps = steps,
        resume_state = resume_state, state_checkpoint_interval = 16,
        state_checkpoint_callback = callback)
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
        seconds = result.training_seconds,
        evaluation_seconds = (time_ns() - evaluation_started) / 1e9,
        center_searches = result.center_global_searches,
        perturbed_searches = result.perturbed_global_searches,
        branch_solves = result.complete_branch_solves,
        relaxation_solves = result.center_relaxation_solves + result.lazy_relaxation_solves,
        observed_threads = result.sample_threads_used)
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
    selection_path = joinpath(directory, "phase18_selection.jls")
    if isfile(selection_path)
        frozen = open(deserialize, selection_path)
        cache_source_compatible(frozen.source) ||
            error("Frozen selection source mismatch.")
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
    save_atomic(selection_path, (schema = PHASE18_SCHEMA,
        source = source_fingerprint(), selected = selected))
    write_json(joinpath(directory, "phase18_selection.json"),
        (schema = PHASE18_SCHEMA, development_seeds = PHASE18_DEVELOPMENT_SEEDS,
            validation_steps = PHASE18_TUNING_STEPS,
            validation_pool_seed = 18_020_002, candidates_per_method = 9,
            total_validation_training_seconds = sum(row.seconds for row in rows),
            total_validation_perturbed_searches = sum(row.perturbed_searches for row in rows),
            source = source_fingerprint(), scores = scores))
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
    rows = filter(row -> row.tag == "fixed", rows)
    complete = length(rows) == length(PHASE18_CONFIRMATORY_SEEDS) * length(PHASE18_LABELS)
    comparisons = Dict{String, Any}()
    if complete
        row_for(seed, label, tag) = only(filter(row -> row.seed == seed &&
            row.method == string(label) && row.tag == tag, rows))
        # Report every paired quality contrast transparently. DBB/DPO remain
        # the registered literature-family comparisons; the others diagnose
        # when the extra SAGE machinery is or is not useful.
        for label in (:hard_local, :ste, :compact_blackbox, :dbb_projection, :dpo_gaussian)
            values = [row_for(seed, :sage_n4, "fixed").ood_loss -
                row_for(seed, label, "fixed").ood_loss for seed in PHASE18_CONFIRMATORY_SEEDS]
            raw = interval(values)
            comparisons[string(label)] = (difference = raw.mean,
                paired_ci95 = [raw.lower, raw.upper],
                lower_is_better = true)
        end
    end
    aggregates = Dict{String, Any}()
    for label in PHASE18_LABELS
        selected = filter(row -> row.method == string(label), rows)
        isempty(selected) && continue
        aggregates["$(label)/fixed"] = (runs = length(selected),
            mean_iid_loss = mean(row.iid_loss for row in selected),
            mean_ood_loss = mean(row.ood_loss for row in selected),
            mean_seconds = mean(row.seconds for row in selected),
            total_perturbed_searches = sum(row.perturbed_searches for row in selected),
            total_branch_solves = sum(row.branch_solves for row in selected))
    end
    summary = (phase = "phase18_external_baselines", schema = PHASE18_SCHEMA,
        source = source_fingerprint(), git_commit = repository_revision(),
        status = complete ? "COMPLETE" : "PARTIAL",
        protocol = "fixed_1024_updates", seeds = PHASE18_CONFIRMATORY_SEEDS,
        fixed_steps = 1024,
        safety_gate = all(row -> row.csat == 1.0 && row.maximum_violation <= 1e-6, rows),
        comparisons = comparisons, aggregates = aggregates,
        baseline_scope = "DBB quadratic-projection adaptation; DPO-inspired Gaussian hard-forward VJP",
        external_dataset = false)
    write_csv(joinpath(directory, "phase18_runs.csv"), rows)
    write_json(joinpath(directory, "phase18_summary.json"), summary)
    return summary
end

function pilot(directory, selected)
    problem, pretrained = EB.setup_problem(1)
    rows = NamedTuple[]
    for label in PHASE18_LABELS
        EB.train_external(problem, pretrained, selected[label], 1; steps = 1)
        push!(rows, execute_method(directory, problem, pretrained, selected[label], 1;
            steps = 256, tag = "pilot"))
    end
    write_csv(joinpath(directory, "phase18_pilot.csv"), rows)
    record = (schema = PHASE18_SCHEMA, source = source_fingerprint(),
        selected = selected, rows = rows)
    save_atomic(joinpath(directory, "phase18_pilot_plan.jls"), record)
    write_json(joinpath(directory, "phase18_pilot_plan.json"), record)
    println("PHASE18_EXTERNAL_PILOT=COMPLETE")
    return record
end

function checked_selection_and_pilot(directory)
    selection_path = joinpath(directory, "phase18_selection.jls")
    pilot_path = joinpath(directory, "phase18_pilot_plan.jls")
    isfile(selection_path) || error("Run the Phase 18 prepare stage first (selection missing).")
    isfile(pilot_path) || error("Run the Phase 18 prepare stage first (pilot missing).")
    selection = open(deserialize, selection_path)
    pilot_record = open(deserialize, pilot_path)
    selection.schema == PHASE18_SCHEMA && cache_source_compatible(selection.source) ||
        error("Frozen selection source/schema mismatch.")
    cache_source_compatible(pilot_record.source) && pilot_record.schema == PHASE18_SCHEMA ||
        error("Pilot plan source/schema mismatch.")
    selection.selected == pilot_record.selected || error("Selection/pilot mismatch.")
    return selection.selected
end

function collect_confirmatory_rows(directory)
    cache_directory = joinpath(directory, "completed_methods")
    isdir(cache_directory) || return NamedTuple[]
    source = source_fingerprint()
    rows = NamedTuple[]
    seen = Set{Tuple{Int, String, String}}()
    for path in sort(readdir(cache_directory; join = true))
        endswith(path, ".jls") || continue
        payload = open(deserialize, path)
        hasproperty(payload, :signature) && hasproperty(payload, :row) || continue
        signature, row = payload.signature, payload.row
        signature.schema == PHASE18_SCHEMA && cache_source_compatible(signature.source, source) || continue
        row.seed in PHASE18_CONFIRMATORY_SEEDS || continue
        valid = row.tag == "fixed" && row.method in string.(PHASE18_LABELS)
        valid || continue
        signature.seed == row.seed && string(signature.spec.label) == row.method &&
            signature.tag == row.tag || error("Cache signature/row mismatch: $(path)")
        key = (row.seed, row.method, row.tag)
        key in seen && error("Duplicate confirmatory cache row: $(key)")
        push!(seen, key)
        push!(rows, row)
    end
    sort!(rows; by = row -> (row.seed, row.tag, row.method))
    return rows
end

function prepare(directory)
    selected = tune(directory)
    pilot(directory, selected)
    println("PHASE18_EXTERNAL_PREPARE=COMPLETE")
end

function confirmatory_seed(directory, seed)
    seed in PHASE18_CONFIRMATORY_SEEDS || error("Seed must be one of 11:15.")
    selected = checked_selection_and_pilot(directory)
    problem, pretrained = EB.setup_problem(seed; pool = :heldout)
    for label in PHASE18_LABELS
        EB.train_external(problem, pretrained, selected[label], seed; steps = 1)
    end
    for label in circshift(collect(PHASE18_LABELS), -mod(seed, length(PHASE18_LABELS)))
        execute_method(directory, problem, pretrained, selected[label], seed;
            steps = 1024, tag = "fixed")
    end
    println("PHASE18_CONFIRM_SEED=COMPLETE")
    return (seed = seed, fixed_rows = length(collect_confirmatory_rows(directory)))
end

function merge_confirmatory(directory)
    checked_selection_and_pilot(directory)
    rows = collect_confirmatory_rows(directory)
    summary = report(directory, rows)
    summary.status == "COMPLETE" || error(
        "Phase 18 merge requires all 30 fixed-step rows; found $(length(rows)).")
    println("method/budget\tmean OOD loss\tmean seconds\tperturbed searches")
    for (name, aggregate) in sort(collect(summary.aggregates); by = first)
        println("$(name)\t$(aggregate.mean_ood_loss)\t$(aggregate.mean_seconds)\t" *
            "$(aggregate.total_perturbed_searches)")
    end
    println(json(summary))
    println("PHASE18_EXTERNAL_CONFIRMATORY=COMPLETE")
    return summary
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE18_PROFILE", "smoke")))
    directory = abspath(get(ENV, "SAGE_PHASE18_OUTPUT",
        joinpath(@__DIR__, "..", "artifacts", "local_runs", "phase18_external_$(profile)")))
    mkpath(directory)
    # Avoid nested BLAS pools around independent solver/sample work.
    BLAS.set_num_threads(1)
    profile == :smoke && return smoke(directory)
    profile == :prepare && return prepare(directory)
    if profile == :confirm_seed
        seed = parse(Int, get(ENV, "SAGE_PHASE18_SEED", "0"))
        return confirmatory_seed(directory, seed)
    end
    profile == :merge && return merge_confirmatory(directory)
    error("SAGE_PHASE18_PROFILE must be smoke, prepare, confirm_seed, or merge.")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
