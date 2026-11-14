using DisjunctiveNet
using Statistics

include(joinpath(@__DIR__, "phase10_jobshop_common.jl"))
using .Phase10JobShop

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
                   for (key, item) in value]
        return "{" * join(entries, ',') * "}"
    end
    error("Unsupported JSON value $(typeof(value)).")
end

function configuration(profile)
    if profile == :smoke
        return (
            seeds = [1], horizons = [2, 4],
            train_size = 4, iid_size = 4, ood_size = 4,
            pretrain_epochs = 2, pretrain_batch = 2,
            radius = 0.5, learning_rate = 1e-3, audit_samples = 4,
            evaluation_interval = 2,
        )
    elseif profile == :pilot
        return (
            seeds = collect(1:5), horizons = [16, 64, 256, 1024],
            train_size = 16, iid_size = 12, ood_size = 12,
            pretrain_epochs = 8, pretrain_batch = 8,
            radius = 0.5, learning_rate = 1e-3, audit_samples = 8,
            evaluation_interval = 64,
        )
    end
    throw(ArgumentError("SAGE_PHASE16_PROFILE must be smoke or pilot."))
end

function subset_split(split::JobShopSplit, indices)
    members = Int.(collect(indices))
    isempty(members) && throw(ArgumentError("Phase 16 strata must be nonempty."))
    return JobShopSplit(
        split.raw_inputs[:, members], split.inputs[:, members],
        split.targets[:, members], split.instances[members],
    )
end

function evaluate_with_strata(network, problem, partition)
    iid = evaluate_network(network, problem.iid; config = problem.config)
    ood = evaluate_network(network, problem.ood; config = problem.config)
    lower = evaluate_network(
        network, subset_split(problem.ood, partition.lower); config = problem.config,
    )
    higher = evaluate_network(
        network, subset_split(problem.ood, partition.higher); config = problem.config,
    )
    return (iid = iid, ood = ood, lower = lower, higher = higher)
end

function paired_interval(values)
    length(values) >= 5 || return (mean = mean(values), lower = NaN, upper = NaN)
    center = mean(values)
    halfwidth = 2.776 * std(values; corrected = true) / sqrt(length(values))
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function method_spec(label)
    label == :hard_local && return (backend = :hard_local, samples = 2)
    label == :ste && return (backend = :ste, samples = 2)
    label == :sage_n4 && return (backend = :sage_multiscale, samples = 4)
    label == :sage_n32 && return (backend = :sage_multiscale, samples = 32)
    label == :sage_adaptive && return (backend = :sage_adaptive, samples = 4)
    throw(ArgumentError("Unknown Phase 16 method $(label)."))
end

function row_for(rows, label, seed, horizon)
    return only(filter(
        row -> row.method == string(label) && row.seed == seed &&
            row.horizon == horizon,
        rows,
    ))
end

function summarize(rows, label, horizon)
    selected = filter(
        row -> row.method == string(label) && row.horizon == horizon,
        rows,
    )
    isempty(selected) && error("Missing Phase 16 $(label)/$(horizon) rows.")
    finite_ess = filter(isfinite, [row.mean_ess for row in selected])
    return (
        iid_loss = mean(row.iid_loss for row in selected),
        ood_loss = mean(row.ood_loss for row in selected),
        lower_loss = mean(row.lower_switch_loss for row in selected),
        higher_loss = mean(row.higher_switch_loss for row in selected),
        elapsed = mean(row.elapsed_seconds for row in selected),
        searches = sum(row.global_searches for row in selected),
        branch_solves = sum(row.branch_solves for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
    )
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE16_PROFILE", "smoke")))
    setup = configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE16_OUTPUT", "phase16_long_horizon_smoke_results",
    ))
    mkpath(output_directory)
    labels = (:hard_local, :ste, :sage_n4, :sage_n32, :sage_adaptive)
    policy = AdaptiveSAGEPolicy()
    rows = NamedTuple[]
    trajectories = NamedTuple[]
    audit_rows = NamedTuple[]
    maximum_horizon = maximum(setup.horizons)
    run_path = joinpath(output_directory, "phase16_checkpoints.csv")
    trajectory_path = joinpath(output_directory, "phase16_trajectories.csv")
    audit_path = joinpath(output_directory, "phase16_switching_audit.csv")
    progress_path = joinpath(output_directory, "phase16_progress.json")
    summary_path = joinpath(output_directory, "phase16_summary.json")

    warm_config = JobShopConfig(n_jobs = 3, n_machines = 4)
    warm = prepare_problem(
        1_016; config = warm_config, train_size = 2, iid_size = 2, ood_size = 2,
    )
    warm_network = pretrain_network(warm, 1_016; epochs = 1, batch_size = 2)
    for label in labels
        spec = method_spec(label)
        warm_result = train_method(
            warm, warm_network, spec.backend, 1_016;
            max_steps = 1, samples = spec.samples, radius = setup.radius,
            evaluation_interval = 1, adaptive_policy = policy,
            checkpoint_steps = [1],
        )
        haskey(warm_result.checkpoint_networks, 1) || error(
            "Phase 16 checkpoint warmup failed for $(label).",
        )
    end

    for seed in setup.seeds
        config = JobShopConfig(n_jobs = 3, n_machines = 4)
        problem = prepare_problem(
            seed;
            config = config, train_size = setup.train_size,
            iid_size = setup.iid_size, ood_size = setup.ood_size,
        )
        pretrained = pretrain_network(
            problem, seed;
            epochs = setup.pretrain_epochs, batch_size = setup.pretrain_batch,
            learning_rate = 2e-3,
        )
        partition = switching_exposure_partition(
            pretrained, problem.ood, 16_000 + seed;
            radius = setup.radius, samples = setup.audit_samples,
        )
        lower_set = Set(partition.lower)
        for index in eachindex(partition.scores)
            push!(audit_rows, (
                seed = seed, instance = index, switch_rate = partition.scores[index],
                stratum = index in lower_set ? "lower_switch" : "higher_switch",
            ))
        end

        shift = mod(seed, length(labels))
        execution_order = circshift(collect(labels), -shift)
        for label in execution_order
            spec = method_spec(label)
            result = train_method(
                problem, pretrained, spec.backend, seed;
                epochs = cld(maximum_horizon, setup.train_size),
                max_steps = maximum_horizon,
                samples = spec.samples,
                radius = setup.radius,
                learning_rate = setup.learning_rate,
                evaluation_interval = setup.evaluation_interval,
                adaptive_policy = policy,
                checkpoint_steps = setup.horizons,
            )
            for item in result.trajectory
                push!(trajectories, (
                    seed = seed, method = string(label), step = item.step,
                    elapsed_seconds = item.elapsed_seconds,
                    iid_task_loss = item.iid_task_loss,
                    ood_task_loss = item.ood_task_loss,
                    global_searches = item.perturbed_global_searches,
                    branch_solves = item.complete_branch_solves,
                    mean_ess = item.mean_effective_sample_size,
                ))
            end
            for horizon in setup.horizons
                network = result.checkpoint_networks[horizon]
                trace = only(filter(item -> item.step == horizon, result.trajectory))
                evaluation = evaluate_with_strata(network, problem, partition)
                maximum_violation = maximum((
                    evaluation.iid.maximum_violation,
                    evaluation.ood.maximum_violation,
                    evaluation.lower.maximum_violation,
                    evaluation.higher.maximum_violation,
                ))
                push!(rows, (
                    seed = seed, method = string(label), horizon = horizon,
                    exact_epochs = horizon / setup.train_size,
                    iid_loss = evaluation.iid.task_loss,
                    iid_csat = evaluation.iid.csat,
                    ood_loss = evaluation.ood.task_loss,
                    ood_csat = evaluation.ood.csat,
                    lower_switch_loss = evaluation.lower.task_loss,
                    higher_switch_loss = evaluation.higher.task_loss,
                    maximum_violation = maximum_violation,
                    elapsed_seconds = trace.elapsed_seconds,
                    global_searches = trace.perturbed_global_searches,
                    branch_solves = trace.complete_branch_solves,
                    switching_samples = trace.switching_samples,
                    mean_ess = trace.mean_effective_sample_size,
                    unresolved = trace.unresolved_samples +
                        evaluation.iid.unresolved + evaluation.ood.unresolved,
                    logical_failures = trace.logical_failures,
                    stable_calls = trace.adaptive_stable_calls,
                    probed_calls = trace.adaptive_probed_calls,
                    probe_searches = trace.adaptive_probe_searches,
                    probe_switches = trace.adaptive_probe_switches,
                    minimum_calls = trace.adaptive_minimum_calls,
                    medium_calls = trace.adaptive_medium_calls,
                    maximum_calls = trace.adaptive_maximum_calls,
                    estimation_samples = trace.adaptive_estimation_samples,
                ))
            end
            println(
                "[phase16] seed=$(seed) method=$(label) steps=$(maximum_horizon) " *
                "seconds=$(round(result.training_seconds; digits = 3)) " *
                "searches=$(result.perturbed_global_searches)",
            )
            flush(stdout)
            write_csv(run_path, propertynames(first(rows)), rows)
            write_csv(trajectory_path, propertynames(first(trajectories)), trajectories)
            write_csv(audit_path, propertynames(first(audit_rows)), audit_rows)
            progress = Dict{String, Any}(
                "profile" => string(profile),
                "last_completed_seed" => seed,
                "last_completed_method" => string(label),
                "completed_method_runs" => length(rows) ÷ length(setup.horizons),
                "target_method_runs" => length(setup.seeds) * length(labels),
                "completed_checkpoint_rows" => length(rows),
                "target_checkpoint_rows" => length(setup.seeds) * length(labels) *
                    length(setup.horizons),
            )
            open(progress_path, "w") do io
                println(io, json_value(progress))
            end
        end
    end

    write_csv(run_path, propertynames(first(rows)), rows)
    write_csv(trajectory_path, propertynames(first(trajectories)), trajectories)
    write_csv(audit_path, propertynames(first(audit_rows)), audit_rows)

    expected_rows = length(setup.seeds) * length(labels) * length(setup.horizons)
    completion_gate = length(rows) == expected_rows
    safety_gate = all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.logical_failures == 0 && row.maximum_violation <= 1e-6
        for row in rows
    )
    monotone_gate = true
    for seed in setup.seeds, label in labels
        selected = sort(
            filter(row -> row.seed == seed && row.method == string(label), rows);
            by = row -> row.horizon,
        )
        monotone_gate &= [row.horizon for row in selected] == setup.horizons
        for index in 2:length(selected)
            previous, current = selected[index - 1], selected[index]
            monotone_gate &= current.elapsed_seconds >= previous.elapsed_seconds &&
                current.global_searches >= previous.global_searches &&
                current.branch_solves >= previous.branch_solves
        end
    end
    sage_gate = all(
        !(row.method in ("sage_n4", "sage_n32", "sage_adaptive")) ||
            (isfinite(row.mean_ess) && row.mean_ess > 0)
        for row in rows
    )
    adaptive_rows = filter(row -> row.method == "sage_adaptive", rows)
    accounting_gate = all(
        row.stable_calls + row.probed_calls == row.horizon &&
        row.probe_searches == policy.probe_samples * row.probed_calls &&
        row.minimum_calls + row.medium_calls + row.maximum_calls == row.horizon &&
        row.estimation_samples ==
            policy.minimum_samples * row.minimum_calls +
            policy.medium_samples * row.medium_calls +
            policy.maximum_samples * row.maximum_calls
        for row in adaptive_rows
    )
    final_gate = all(
        any(row -> row.seed == seed && row.method == string(label) &&
            row.horizon == maximum_horizon, rows)
        for seed in setup.seeds, label in labels
    )

    results = Dict{String, Any}()
    for horizon in setup.horizons
        horizon_results = Dict{String, Any}()
        for label in labels
            item = summarize(rows, label, horizon)
            horizon_results[string(label)] = Dict(
                "iid_loss" => item.iid_loss,
                "ood_loss" => item.ood_loss,
                "lower_switch_loss" => item.lower_loss,
                "higher_switch_loss" => item.higher_loss,
                "elapsed_seconds" => item.elapsed,
                "global_searches" => item.searches,
                "branch_solves" => item.branch_solves,
                "mean_ess" => item.ess,
            )
        end
        results[string(horizon)] = horizon_results
    end

    sage_minus_ste = paired_interval([
        row_for(rows, :sage_n32, seed, maximum_horizon).ood_loss -
        row_for(rows, :ste, seed, maximum_horizon).ood_loss
        for seed in setup.seeds
    ])
    sage_minus_hard = paired_interval([
        row_for(rows, :sage_n32, seed, maximum_horizon).ood_loss -
        row_for(rows, :hard_local, seed, maximum_horizon).ood_loss
        for seed in setup.seeds
    ])
    final_sage = summarize(rows, :sage_n32, maximum_horizon)
    final_ste = summarize(rows, :ste, maximum_horizon)
    final_hard = summarize(rows, :hard_local, maximum_horizon)
    crossover = final_sage.ood_loss < final_ste.ood_loss &&
        final_sage.ood_loss < final_hard.ood_loss &&
        (sage_minus_ste.upper < 0 || sage_minus_hard.upper < 0)
    crossover_status = crossover ? "OBSERVED" : "NOT_OBSERVED"

    lower_audit = filter(row -> row.stratum == "lower_switch", audit_rows)
    higher_audit = filter(row -> row.stratum == "higher_switch", audit_rows)
    phase16_pass = profile == :pilot && completion_gate && safety_gate &&
        monotone_gate && sage_gate && accounting_gate && final_gate
    summary = Dict{String, Any}(
        "phase" => "phase16_long_horizon",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" :
            phase16_pass ? "PASS" : "FAIL",
        "crossover_status" => crossover_status,
        "seeds" => setup.seeds,
        "horizons" => setup.horizons,
        "exact_epochs" => [horizon / setup.train_size for horizon in setup.horizons],
        "results" => results,
        "sage_n32_minus_ste_ci95" => [
            sage_minus_ste.mean, sage_minus_ste.lower, sage_minus_ste.upper,
        ],
        "sage_n32_minus_hard_ci95" => [
            sage_minus_hard.mean, sage_minus_hard.lower, sage_minus_hard.upper,
        ],
        "switching_audit_lower_rate" => mean(row.switch_rate for row in lower_audit),
        "switching_audit_higher_rate" => mean(row.switch_rate for row in higher_audit),
        "completion_gate" => completion_gate,
        "safety_gate" => safety_gate,
        "monotone_work_gate" => monotone_gate,
        "sage_ess_gate" => sage_gate,
        "adaptive_accounting_gate" => accounting_gate,
        "final_checkpoint_gate" => final_gate,
        "pass" => phase16_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 16 long-horizon convergence study ", profile)
    println("Julia version: ", VERSION)
    println("Contract: uninterrupted training; exact checkpoints; fixed rho/backwards")
    println("horizon\tepochs\tmethod\tOOD loss\thigher-switch loss\tseconds\tsearches\tbranch solves")
    for horizon in setup.horizons, label in labels
        item = summarize(rows, label, horizon)
        println("$(horizon)\t$(horizon / setup.train_size)\t$(label)\t" *
            "$(item.ood_loss)\t$(item.higher_loss)\t$(item.elapsed)\t" *
            "$(item.searches)\t$(item.branch_solves)")
    end
    println("final SAGE N=32 - STE OOD loss/95% CI: ", sage_minus_ste.mean,
        " / [", sage_minus_ste.lower, ", ", sage_minus_ste.upper, "]")
    println("final SAGE N=32 - hard-local OOD loss/95% CI: ", sage_minus_hard.mean,
        " / [", sage_minus_hard.lower, ", ", sage_minus_hard.upper, "]")
    println("long-horizon crossover: ", crossover_status)
    for (name, passed) in (
        ("completion", completion_gate), ("safety", safety_gate),
        ("monotone work", monotone_gate), ("SAGE ESS", sage_gate),
        ("adaptive accounting", accounting_gate),
        ("final checkpoint", final_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("checkpoint CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("switching audit CSV: ", audit_path)
    println("progress JSON: ", progress_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE16_LONG_HORIZON_SMOKE=COMPLETE" :
        "PHASE16_LONG_HORIZON=" * (phase16_pass ? "PASS" : "FAIL"))
    profile == :pilot && println("PHASE16_CROSSOVER=", crossover_status)
end

main()
