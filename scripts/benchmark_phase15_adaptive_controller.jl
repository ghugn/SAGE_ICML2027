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
            seeds = [1], train_size = 4, iid_size = 4, ood_size = 4,
            pretrain_epochs = 2, pretrain_batch = 2, steps = 2,
            radius = 0.5, learning_rate = 1e-3, audit_samples = 4,
        )
    elseif profile == :pilot
        return (
            seeds = collect(1:5), train_size = 16, iid_size = 12, ood_size = 12,
            pretrain_epochs = 8, pretrain_batch = 8, steps = 12,
            radius = 0.5, learning_rate = 1e-3, audit_samples = 8,
        )
    end
    throw(ArgumentError("SAGE_PHASE15_PROFILE must be smoke or pilot."))
end

function subset_split(split::JobShopSplit, indices)
    members = Int.(collect(indices))
    isempty(members) && throw(ArgumentError("Phase 15 strata must be nonempty."))
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

function normalized_auc(trajectory)
    length(trajectory) >= 2 || return trajectory[1].ood_task_loss
    area = 0.0
    for index in 2:length(trajectory)
        left, right = trajectory[index - 1], trajectory[index]
        area += (right.step - left.step) *
            (left.ood_task_loss + right.ood_task_loss) / 2
    end
    width = trajectory[end].step - trajectory[1].step
    return width > 0 ? area / width : trajectory[end].ood_task_loss
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
    throw(ArgumentError("Unknown Phase 15 method $(label)."))
end

function result_row(seed, label, result, evaluation)
    return (
        seed = seed, method = string(label), steps = result.steps,
        iid_loss = evaluation.iid.task_loss, iid_csat = evaluation.iid.csat,
        ood_loss = evaluation.ood.task_loss, ood_csat = evaluation.ood.csat,
        ood_auc = normalized_auc(result.trajectory),
        lower_switch_loss = evaluation.lower.task_loss,
        higher_switch_loss = evaluation.higher.task_loss,
        maximum_violation = maximum((
            evaluation.iid.maximum_violation, evaluation.ood.maximum_violation,
            evaluation.lower.maximum_violation, evaluation.higher.maximum_violation,
        )),
        seconds = result.training_seconds,
        global_searches = result.perturbed_global_searches,
        branch_solves = result.complete_branch_solves,
        switching_samples = result.switching_samples,
        certificates = result.sample_certificates,
        unresolved = result.unresolved_samples + evaluation.iid.unresolved +
            evaluation.ood.unresolved,
        logical_failures = result.logical_failures,
        mean_ess = result.mean_effective_sample_size,
        stable_calls = result.adaptive_stable_calls,
        probed_calls = result.adaptive_probed_calls,
        probe_searches = result.adaptive_probe_searches,
        probe_switches = result.adaptive_probe_switches,
        minimum_calls = result.adaptive_minimum_calls,
        medium_calls = result.adaptive_medium_calls,
        maximum_calls = result.adaptive_maximum_calls,
        estimation_samples = result.adaptive_estimation_samples,
    )
end

function summarize(rows, label)
    selected = filter(row -> row.method == string(label), rows)
    isempty(selected) && error("Missing Phase 15 method $(label).")
    finite_ess = filter(isfinite, [row.mean_ess for row in selected])
    return (
        ood_loss = mean(row.ood_loss for row in selected),
        ood_auc = mean(row.ood_auc for row in selected),
        lower_loss = mean(row.lower_switch_loss for row in selected),
        higher_loss = mean(row.higher_switch_loss for row in selected),
        seconds = mean(row.seconds for row in selected),
        searches = sum(row.global_searches for row in selected),
        branch_solves = sum(row.branch_solves for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
        steps = sum(row.steps for row in selected),
        stable_calls = sum(row.stable_calls for row in selected),
        probed_calls = sum(row.probed_calls for row in selected),
        probe_searches = sum(row.probe_searches for row in selected),
        probe_switches = sum(row.probe_switches for row in selected),
        minimum_calls = sum(row.minimum_calls for row in selected),
        medium_calls = sum(row.medium_calls for row in selected),
        maximum_calls = sum(row.maximum_calls for row in selected),
        estimation_samples = sum(row.estimation_samples for row in selected),
    )
end

row_for(rows, label, seed) = only(filter(
    row -> row.method == string(label) && row.seed == seed,
    rows,
))

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE15_PROFILE", "smoke")))
    setup = configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE15_OUTPUT", "phase15_adaptive_smoke_results",
    ))
    mkpath(output_directory)
    policy = AdaptiveSAGEPolicy(
        probe_samples = 4,
        minimum_samples = 4,
        medium_samples = 16,
        maximum_samples = 32,
        medium_switch_threshold = 1,
        maximum_switch_threshold = 2,
    )
    labels = (:hard_local, :ste, :sage_n4, :sage_n32, :sage_adaptive)
    rows = NamedTuple[]
    trajectories = NamedTuple[]
    audit_rows = NamedTuple[]

    warm_config = JobShopConfig(n_jobs = 3, n_machines = 4)
    warm = prepare_problem(
        999; config = warm_config, train_size = 2, iid_size = 2, ood_size = 2,
    )
    warm_network = pretrain_network(warm, 999; epochs = 1, batch_size = 2)
    for label in labels
        spec = method_spec(label)
        train_method(
            warm, warm_network, spec.backend, 999;
            max_steps = 1, samples = spec.samples, radius = setup.radius,
            evaluation_interval = 1, adaptive_policy = policy,
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
            pretrained, problem.ood, 15_000 + seed;
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
                epochs = cld(setup.steps, setup.train_size),
                max_steps = setup.steps,
                samples = spec.samples,
                radius = setup.radius,
                learning_rate = setup.learning_rate,
                evaluation_interval = min(4, setup.steps),
                adaptive_policy = policy,
            )
            evaluation = evaluate_with_strata(result.network, problem, partition)
            row = result_row(seed, label, result, evaluation)
            push!(rows, row)
            for item in result.trajectory
                push!(trajectories, (
                    seed = seed, method = string(label), step = item.step,
                    iid_task_loss = item.iid_task_loss,
                    ood_task_loss = item.ood_task_loss,
                ))
            end
            println(
                "[phase15] seed=$(seed) method=$(label) " *
                "seconds=$(round(result.training_seconds; digits = 3)) " *
                "searches=$(result.perturbed_global_searches)",
            )
            flush(stdout)
        end
    end

    run_path = joinpath(output_directory, "phase15_runs.csv")
    trajectory_path = joinpath(output_directory, "phase15_trajectories.csv")
    audit_path = joinpath(output_directory, "phase15_switching_audit.csv")
    summary_path = joinpath(output_directory, "phase15_summary.json")
    write_csv(run_path, propertynames(first(rows)), rows)
    write_csv(trajectory_path, propertynames(first(trajectories)), trajectories)
    write_csv(audit_path, propertynames(first(audit_rows)), audit_rows)

    expected_rows = length(setup.seeds) * length(labels)
    completion_gate = length(rows) == expected_rows
    safety_gate = all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.logical_failures == 0 && row.maximum_violation <= 1e-6
        for row in rows
    )
    adaptive_rows = filter(row -> row.method == "sage_adaptive", rows)
    accounting_gate = all(
        row.stable_calls + row.probed_calls == row.steps &&
        row.probe_searches == policy.probe_samples * row.probed_calls &&
        row.minimum_calls + row.medium_calls + row.maximum_calls == row.steps &&
        row.estimation_samples ==
            policy.minimum_samples * row.minimum_calls +
            policy.medium_samples * row.medium_calls +
            policy.maximum_samples * row.maximum_calls
        for row in adaptive_rows
    )

    method_summaries = Dict{String, Any}()
    for label in labels
        item = summarize(rows, label)
        method_summaries[string(label)] = Dict(
            "ood_loss" => item.ood_loss,
            "ood_auc" => item.ood_auc,
            "lower_switch_loss" => item.lower_loss,
            "higher_switch_loss" => item.higher_loss,
            "seconds" => item.seconds,
            "global_searches" => item.searches,
            "branch_solves" => item.branch_solves,
            "mean_ess" => item.ess,
            "steps" => item.steps,
        )
    end
    adaptive = summarize(rows, :sage_adaptive)
    fixed_low = summarize(rows, :sage_n4)
    fixed_high = summarize(rows, :sage_n32)
    quality_ratio = adaptive.ood_loss / min(fixed_low.ood_loss, fixed_high.ood_loss)
    search_ratio = adaptive.searches / max(fixed_high.searches, 1)
    quality_gate = quality_ratio <= 1.10
    compute_gate = adaptive.searches <= fixed_high.searches
    joint_utility_gate = quality_gate && compute_gate

    adaptive_minus_ste = paired_interval([
        row_for(rows, :sage_adaptive, seed).ood_loss -
        row_for(rows, :ste, seed).ood_loss for seed in setup.seeds
    ])
    adaptive_minus_hard = paired_interval([
        row_for(rows, :sage_adaptive, seed).ood_loss -
        row_for(rows, :hard_local, seed).ood_loss for seed in setup.seeds
    ])
    adaptive_minus_n32 = paired_interval([
        row_for(rows, :sage_adaptive, seed).ood_loss -
        row_for(rows, :sage_n32, seed).ood_loss for seed in setup.seeds
    ])
    lower_audit = filter(row -> row.stratum == "lower_switch", audit_rows)
    higher_audit = filter(row -> row.stratum == "higher_switch", audit_rows)

    phase15_pass = profile == :pilot && completion_gate && safety_gate &&
        accounting_gate && joint_utility_gate
    summary = Dict{String, Any}(
        "phase" => "phase15_adaptive_controller",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" :
            phase15_pass ? "PASS" : "FAIL",
        "policy" => Dict(
            "probe_samples" => policy.probe_samples,
            "minimum_samples" => policy.minimum_samples,
            "medium_samples" => policy.medium_samples,
            "maximum_samples" => policy.maximum_samples,
            "medium_switch_threshold" => policy.medium_switch_threshold,
            "maximum_switch_threshold" => policy.maximum_switch_threshold,
        ),
        "methods" => method_summaries,
        "quality_ratio_vs_best_fixed" => quality_ratio,
        "search_ratio_vs_fixed_n32" => search_ratio,
        "adaptive_stable_calls" => adaptive.stable_calls,
        "adaptive_probed_calls" => adaptive.probed_calls,
        "adaptive_probe_switches" => adaptive.probe_switches,
        "adaptive_minimum_calls" => adaptive.minimum_calls,
        "adaptive_medium_calls" => adaptive.medium_calls,
        "adaptive_maximum_calls" => adaptive.maximum_calls,
        "adaptive_estimation_samples" => adaptive.estimation_samples,
        "switching_audit_lower_rate" => mean(row.switch_rate for row in lower_audit),
        "switching_audit_higher_rate" => mean(row.switch_rate for row in higher_audit),
        "adaptive_minus_ste_ci95" => [
            adaptive_minus_ste.mean, adaptive_minus_ste.lower, adaptive_minus_ste.upper,
        ],
        "adaptive_minus_hard_ci95" => [
            adaptive_minus_hard.mean, adaptive_minus_hard.lower, adaptive_minus_hard.upper,
        ],
        "adaptive_minus_n32_ci95" => [
            adaptive_minus_n32.mean, adaptive_minus_n32.lower,
            adaptive_minus_n32.upper,
        ],
        "completion_gate" => completion_gate,
        "safety_gate" => safety_gate,
        "accounting_gate" => accounting_gate,
        "quality_gate" => quality_gate,
        "compute_gate" => compute_gate,
        "joint_utility_gate" => joint_utility_gate,
        "pass" => phase15_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 15 certificate-triggered adaptive controller ", profile)
    println("Julia version: ", VERSION)
    println("Retained contract: exact forward, Eq.(11)-(12),(17)-(21), fixed rho, exact q/p")
    println("New controller: independent 4-probe switch count -> N=4/16/32")
    println("method\tOOD loss\thigher-switch loss\tseconds\tglobal searches\tbranch solves\tESS")
    for label in labels
        item = summarize(rows, label)
        println("$(label)\t$(item.ood_loss)\t$(item.higher_loss)\t" *
            "$(item.seconds)\t$(item.searches)\t$(item.branch_solves)\t$(item.ess)")
    end
    println("adaptive stable/probed calls: ", adaptive.stable_calls, "/", adaptive.probed_calls)
    println("adaptive N=4/16/32 calls: ", adaptive.minimum_calls, "/",
        adaptive.medium_calls, "/", adaptive.maximum_calls)
    println("adaptive probe switches/searches: ", adaptive.probe_switches, "/",
        adaptive.probe_searches)
    println("quality ratio vs best fixed SAGE: ", quality_ratio)
    println("search ratio vs fixed N=32: ", search_ratio)
    println("adaptive - STE OOD loss/95% CI: ", adaptive_minus_ste.mean, " / [",
        adaptive_minus_ste.lower, ", ", adaptive_minus_ste.upper, "]")
    println("adaptive - hard-local OOD loss/95% CI: ", adaptive_minus_hard.mean, " / [",
        adaptive_minus_hard.lower, ", ", adaptive_minus_hard.upper, "]")
    for (name, passed) in (
        ("completion", completion_gate), ("safety", safety_gate),
        ("controller accounting", accounting_gate), ("quality", quality_gate),
        ("compute", compute_gate), ("joint utility", joint_utility_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("switching audit CSV: ", audit_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE15_ADAPTIVE_SMOKE=COMPLETE" :
        "PHASE15_ADAPTIVE_CONTROLLER=" * (phase15_pass ? "PASS" : "FAIL"))
end

main()
