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
            seeds = [1], machines = [3, 4], horizons = [2, 4],
            train_size = 6, iid_size = 4, ood_size = 4,
            pretrain_epochs = 2, pretrain_batch = 3,
            samples = 4, radius = 0.5, learning_rate = 1e-3,
            audit_samples = 4, reference_samples = 4, sage_audit_samples = 4,
            wall_step_cap = 2,
        )
    elseif profile == :pilot
        return (
            seeds = collect(1:5), machines = [3, 4], horizons = [6, 24],
            train_size = 16, iid_size = 12, ood_size = 12,
            pretrain_epochs = 8, pretrain_batch = 8,
            samples = 4, radius = 0.5, learning_rate = 1e-3,
            audit_samples = 8, reference_samples = 64, sage_audit_samples = 32,
            wall_step_cap = 4,
        )
    end
    throw(ArgumentError("SAGE_PHASE13_PROFILE must be smoke or pilot."))
end

function paired_interval(values)
    length(values) >= 5 || return (mean = mean(values), lower = NaN, upper = NaN)
    center = mean(values)
    halfwidth = 2.776 * std(values; corrected = true) / sqrt(length(values))
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function subset_split(split::JobShopSplit, indices)
    members = Int.(collect(indices))
    isempty(members) && throw(ArgumentError("Phase 13 strata must be nonempty."))
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

function result_row(seed, machines, horizon, protocol, method, evaluation, result, budget)
    return (
        seed = Int(seed), jobs = 3, machines = machines,
        disjunctions = machines * binomial(3, 2),
        branches = 2^(machines * binomial(3, 2)),
        horizon = horizon, protocol = string(protocol), method = string(method),
        iid_loss = evaluation.iid.task_loss, iid_csat = evaluation.iid.csat,
        ood_loss = evaluation.ood.task_loss,
        ood_auc = normalized_auc(result.trajectory),
        ood_makespan_gap = evaluation.ood.makespan_gap,
        ood_csat = evaluation.ood.csat,
        lower_switch_loss = evaluation.lower.task_loss,
        higher_switch_loss = evaluation.higher.task_loss,
        maximum_violation = maximum((
            evaluation.iid.maximum_violation, evaluation.ood.maximum_violation,
            evaluation.lower.maximum_violation, evaluation.higher.maximum_violation,
        )),
        seconds = result.training_seconds, planned_budget_seconds = budget,
        steps = result.steps, searches = result.perturbed_global_searches,
        branch_solves = result.complete_branch_solves,
        certificates = result.sample_certificates,
        switching = result.switching_samples,
        unresolved = result.unresolved_samples + evaluation.iid.unresolved +
            evaluation.ood.unresolved,
        logical_failures = result.logical_failures,
        mean_ess = result.mean_effective_sample_size,
        mean_local_radius = result.mean_local_radius,
        positive_local_calls = result.positive_local_radius_calls,
        base_samples = result.base_proposal_samples,
        local_samples = result.local_proposal_samples,
        focused_samples = result.focused_proposal_samples,
    )
end

function summarize(rows, machines, horizon, protocol, method)
    selected = filter(row -> row.machines == machines && row.horizon == horizon &&
        row.protocol == string(protocol) && row.method == string(method), rows)
    isempty(selected) && error("Missing Phase 13 cell $(machines)/$(horizon)/$(protocol)/$(method).")
    finite_ess = filter(isfinite, [row.mean_ess for row in selected])
    return (
        ood_loss = mean(row.ood_loss for row in selected),
        ood_auc = mean(row.ood_auc for row in selected),
        higher_switch_loss = mean(row.higher_switch_loss for row in selected),
        lower_switch_loss = mean(row.lower_switch_loss for row in selected),
        seconds = mean(row.seconds for row in selected),
        steps = sum(row.steps for row in selected),
        searches = sum(row.searches for row in selected),
        branch_solves = sum(row.branch_solves for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
        positive_local_calls = sum(row.positive_local_calls for row in selected),
        local_samples = sum(row.local_samples for row in selected),
    )
end

row_for(rows, machines, horizon, protocol, method, seed) = only(filter(
    row -> row.machines == machines && row.horizon == horizon &&
        row.protocol == string(protocol) && row.method == string(method) &&
        row.seed == seed,
    rows,
))

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE13_PROFILE", "smoke")))
    setup = configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE13_OUTPUT", "phase13_difficulty_map_smoke_results",
    ))
    mkpath(output_directory)
    methods = (:hard_local, :ste, :sage_multiscale)
    rows = NamedTuple[]
    trajectories = NamedTuple[]
    audit_rows = NamedTuple[]
    gradient_rows = NamedTuple[]
    maximum_violation = 0.0
    structure_ok = true
    stratification_ok = true

    # Warm all three paths before timings; the larger dimensional path is also
    # warmed because Julia specialization and solver setup depend on shape.
    for machines in setup.machines
        config = JobShopConfig(n_jobs = 3, n_machines = machines)
        warm = prepare_problem(
            999 + machines; config = config, train_size = 2, iid_size = 2, ood_size = 2,
        )
        warm_network = pretrain_network(warm, 999 + machines; epochs = 1, batch_size = 2)
        for method in methods
            train_method(
                warm, warm_network, method, 999 + machines;
                max_steps = 1, samples = 2, radius = setup.radius,
                evaluation_interval = 1,
            )
        end
    end

    for machines in setup.machines, seed in setup.seeds
        config = JobShopConfig(n_jobs = 3, n_machines = machines)
        expected_disjunctions = machines * binomial(3, 2)
        expected_branches = 2^expected_disjunctions
        problem = prepare_problem(
            seed;
            config = config, train_size = setup.train_size,
            iid_size = setup.iid_size, ood_size = setup.ood_size,
        )
        structure_ok &= all(
            length(instance.model.disjunctions) == expected_disjunctions &&
            num_branches(instance.model) == expected_branches &&
            check_feasibility(instance.model, instance.oracle_starts).feasible
            for split in (problem.train, problem.iid, problem.ood)
            for instance in split.instances
        )
        pretrained = pretrain_network(
            problem, seed;
            epochs = setup.pretrain_epochs, batch_size = setup.pretrain_batch,
            learning_rate = 2e-3,
        )
        partition = switching_exposure_partition(
            pretrained, problem.ood, 10_000machines + seed;
            radius = setup.radius, samples = setup.audit_samples,
        )
        stratification_ok &= !isempty(partition.lower) && !isempty(partition.higher) &&
            partition.higher_rate >= partition.lower_rate
        lower_set = Set(partition.lower)
        for index in eachindex(partition.scores)
            push!(audit_rows, (
                seed = seed, jobs = 3, machines = machines, instance = index,
                switch_rate = partition.scores[index],
                stratum = index in lower_set ? "lower_switch" : "higher_switch",
            ))
        end

        gradient = output_gradient_audit(
            problem, pretrained, 1, 10_000machines + seed;
            radius = setup.radius, reference_samples = setup.reference_samples,
            sage_samples = setup.sage_audit_samples,
        )
        for (method, item) in (
            (:hard_local, gradient.hard_local), (:ste, gradient.ste),
            (:sage_multiscale, gradient.sage_multiscale),
        )
            push!(gradient_rows, (
                seed = seed, jobs = 3, machines = machines, method = string(method),
                cosine = item.cosine, relative_error = item.relative_error,
                reference_norm = gradient.reference_norm,
                reference_samples = setup.reference_samples,
                audit_samples = method == :sage_multiscale ? setup.sage_audit_samples : 0,
                global_searches = item.global_searches,
                branch_solves = item.branch_solves,
                ess = item.ess, unresolved = item.unresolved,
                logical_failures = item.logical_failures,
            ))
        end

        hardest_primary = Dict{Symbol, Any}()
        for horizon in setup.horizons
            epochs = cld(horizon, setup.train_size)
            shift = mod(seed + machines + horizon, length(methods))
            order = circshift(collect(methods), -shift)
            for method in order
                result = train_method(
                    problem, pretrained, method, seed;
                    epochs = epochs, max_steps = horizon, samples = setup.samples,
                    radius = setup.radius, learning_rate = setup.learning_rate,
                    evaluation_interval = min(6, horizon),
                )
                evaluation = evaluate_with_strata(result.network, problem, partition)
                row = result_row(
                    seed, machines, horizon, :matched_steps, method,
                    evaluation, result, 0.0,
                )
                push!(rows, row)
                maximum_violation = max(maximum_violation, row.maximum_violation)
                for item in result.trajectory
                    push!(trajectories, (
                        seed = seed, jobs = 3, machines = machines,
                        horizon = horizon, protocol = "matched_steps",
                        method = string(method), step = item.step,
                        iid_task_loss = item.iid_task_loss,
                        ood_task_loss = item.ood_task_loss,
                    ))
                end
                horizon == maximum(setup.horizons) &&
                    (hardest_primary[method] = result)
                println(
                    "[phase13/matched] size=3x$(machines) seed=$(seed) " *
                    "horizon=$(horizon) method=$(method) " *
                    "seconds=$(round(result.training_seconds; digits = 3))",
                )
                flush(stdout)
            end
        end

        if machines == maximum(setup.machines)
            hardest = maximum(setup.horizons)
            target_seconds = maximum(
                result.training_seconds for result in values(hardest_primary)
            )
            planned_steps = Dict{Symbol, Int}()
            for method in methods
                result = hardest_primary[method]
                seconds_per_step = result.training_seconds / max(result.steps, 1)
                estimated = floor(Int, target_seconds / max(seconds_per_step, eps(Float64)))
                planned_steps[method] = clamp(
                    estimated, hardest, setup.wall_step_cap * hardest,
                )
            end
            for method in reverse(collect(methods))
                steps = planned_steps[method]
                result = train_method(
                    problem, pretrained, method, seed;
                    epochs = cld(steps, setup.train_size), max_steps = steps,
                    samples = setup.samples, radius = setup.radius,
                    learning_rate = setup.learning_rate,
                    evaluation_interval = min(6, steps),
                )
                evaluation = evaluate_with_strata(result.network, problem, partition)
                row = result_row(
                    seed, machines, hardest, :calibrated_time, method,
                    evaluation, result, target_seconds,
                )
                push!(rows, row)
                maximum_violation = max(maximum_violation, row.maximum_violation)
                for item in result.trajectory
                    push!(trajectories, (
                        seed = seed, jobs = 3, machines = machines,
                        horizon = hardest, protocol = "calibrated_time",
                        method = string(method), step = item.step,
                        iid_task_loss = item.iid_task_loss,
                        ood_task_loss = item.ood_task_loss,
                    ))
                end
                println(
                    "[phase13/time] size=3x$(machines) seed=$(seed) " *
                    "method=$(method) steps=$(steps) " *
                    "seconds=$(round(result.training_seconds; digits = 3))",
                )
                flush(stdout)
            end
        end
    end

    run_path = joinpath(output_directory, "phase13_runs.csv")
    trajectory_path = joinpath(output_directory, "phase13_trajectories.csv")
    audit_path = joinpath(output_directory, "phase13_switching_audit.csv")
    gradient_path = joinpath(output_directory, "phase13_gradient_audit.csv")
    summary_path = joinpath(output_directory, "phase13_summary.json")
    write_csv(run_path, propertynames(first(rows)), rows)
    write_csv(trajectory_path, propertynames(first(trajectories)), trajectories)
    write_csv(audit_path, propertynames(first(audit_rows)), audit_rows)
    write_csv(gradient_path, propertynames(first(gradient_rows)), gradient_rows)

    expected_primary = length(setup.seeds) * length(setup.machines) *
        length(setup.horizons) * length(methods)
    expected_time = length(setup.seeds) * length(methods)
    completion_gate = count(row -> row.protocol == "matched_steps", rows) ==
        expected_primary && count(row -> row.protocol == "calibrated_time", rows) ==
        expected_time
    safety_gate = maximum_violation <= 1e-6 && all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.logical_failures == 0 && row.maximum_violation <= 1e-6 for row in rows
    )
    gradient_gate = all(
        isfinite(row.cosine) && isfinite(row.relative_error) &&
        isfinite(row.reference_norm) && row.reference_norm > 0 &&
        row.unresolved == 0 && row.logical_failures == 0 &&
        (row.method != "sage_multiscale" || (isfinite(row.ess) && row.ess > 0))
        for row in gradient_rows
    )

    cell_summaries = Dict{String, Any}()
    estimator_gate = true
    for machines in setup.machines, horizon in setup.horizons
        key = "3x$(machines)_h$(horizon)"
        cell_summaries[key] = Dict{String, Any}()
        for method in methods
            item = summarize(rows, machines, horizon, :matched_steps, method)
            cell_summaries[key][string(method)] = Dict(
                "ood_loss" => item.ood_loss,
                "ood_auc" => item.ood_auc,
                "higher_switch_loss" => item.higher_switch_loss,
                "seconds" => item.seconds,
                "steps" => item.steps,
                "global_searches" => item.searches,
                "branch_solves" => item.branch_solves,
                "ess" => item.ess,
            )
        end
        sage = summarize(rows, machines, horizon, :matched_steps, :sage_multiscale)
        positive_rate = sage.positive_local_calls / max(sage.steps, 1)
        estimator_gate &= sage.ess >= 0.40 * setup.samples &&
            positive_rate >= 0.30 && sage.local_samples > 0
        cell_summaries[key]["sage_positive_local_rate"] = positive_rate
    end

    hardest_machines = maximum(setup.machines)
    hardest_horizon = maximum(setup.horizons)
    hard = summarize(rows, hardest_machines, hardest_horizon, :matched_steps, :hard_local)
    ste = summarize(rows, hardest_machines, hardest_horizon, :matched_steps, :ste)
    sage = summarize(rows, hardest_machines, hardest_horizon, :matched_steps, :sage_multiscale)
    sage_ste = paired_interval([
        row_for(rows, hardest_machines, hardest_horizon, :matched_steps,
            :sage_multiscale, seed).higher_switch_loss -
        row_for(rows, hardest_machines, hardest_horizon, :matched_steps,
            :ste, seed).higher_switch_loss for seed in setup.seeds
    ])
    sage_hard = paired_interval([
        row_for(rows, hardest_machines, hardest_horizon, :matched_steps,
            :sage_multiscale, seed).higher_switch_loss -
        row_for(rows, hardest_machines, hardest_horizon, :matched_steps,
            :hard_local, seed).higher_switch_loss for seed in setup.seeds
    ])
    crossover = sage.higher_switch_loss < ste.higher_switch_loss &&
        sage.higher_switch_loss < hard.higher_switch_loss &&
        (sage_ste.upper < 0 || sage_hard.upper < 0)
    crossover_status = crossover ? "OBSERVED" : "NOT_OBSERVED"

    time_summary = Dict{String, Any}()
    for method in methods
        item = summarize(
            rows, hardest_machines, hardest_horizon, :calibrated_time, method,
        )
        time_summary[string(method)] = Dict(
            "ood_loss" => item.ood_loss,
            "ood_auc" => item.ood_auc,
            "higher_switch_loss" => item.higher_switch_loss,
            "seconds" => item.seconds,
            "steps" => item.steps,
        )
    end

    gradient_summary = Dict{String, Any}()
    switching_summary = Dict{String, Any}()
    for machines in setup.machines
        gradient_summary["3x$(machines)"] = Dict{String, Any}()
        for method in methods
            selected = filter(row -> row.machines == machines &&
                row.method == string(method), gradient_rows)
            gradient_summary["3x$(machines)"][string(method)] = Dict(
                "mean_cosine" => mean(row.cosine for row in selected),
                "mean_relative_error" => mean(row.relative_error for row in selected),
                "mean_ess" => begin
                    values = filter(isfinite, [row.ess for row in selected])
                    isempty(values) ? NaN : mean(values)
                end,
            )
        end
        lower = filter(row -> row.machines == machines &&
            row.stratum == "lower_switch", audit_rows)
        higher = filter(row -> row.machines == machines &&
            row.stratum == "higher_switch", audit_rows)
        switching_summary["3x$(machines)"] = Dict(
            "lower_rate" => mean(row.switch_rate for row in lower),
            "higher_rate" => mean(row.switch_rate for row in higher),
        )
    end

    phase13_pass = profile == :pilot && safety_gate && structure_ok &&
        completion_gate && stratification_ok && gradient_gate && estimator_gate
    summary = Dict{String, Any}(
        "phase" => "phase13_difficulty_map",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" :
            phase13_pass ? "PASS" : "FAIL",
        "crossover_status" => crossover_status,
        "seeds" => setup.seeds, "machines" => setup.machines,
        "horizons" => setup.horizons, "samples" => setup.samples,
        "radius" => setup.radius, "cells" => cell_summaries,
        "calibrated_time_hardest" => time_summary,
        "gradient_audit" => gradient_summary,
        "switching_exposure" => switching_summary,
        "hardest_sage_ste_higher_switch_difference" => sage_ste.mean,
        "hardest_sage_ste_higher_switch_ci95" => [sage_ste.lower, sage_ste.upper],
        "hardest_sage_hard_higher_switch_difference" => sage_hard.mean,
        "hardest_sage_hard_higher_switch_ci95" => [sage_hard.lower, sage_hard.upper],
        "maximum_violation" => maximum_violation,
        "safety_gate" => safety_gate, "structure_gate" => structure_ok,
        "completion_gate" => completion_gate,
        "stratification_gate" => stratification_ok,
        "gradient_gate" => gradient_gate, "estimator_gate" => estimator_gate,
        "pass" => phase13_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 13 pre-registered difficulty map ", profile)
    println("Julia version: ", VERSION)
    println("Contract: exact forward; fixed compact q/rho per slice; Eq.(11)-(12),(17)-(21); exact q/p")
    println("grid machines/horizons/seeds/N: ", setup.machines, "/",
        setup.horizons, "/", length(setup.seeds), "/", setup.samples)
    println("method\tsize\thorizon\tOOD loss\thigher-switch\tAUC\tseconds\tsearches\tbranch solves\tESS")
    for machines in setup.machines, horizon in setup.horizons, method in methods
        item = summarize(rows, machines, horizon, :matched_steps, method)
        println("$(method)\t3x$(machines)\t$(horizon)\t$(item.ood_loss)\t" *
            "$(item.higher_switch_loss)\t$(item.ood_auc)\t$(item.seconds)\t" *
            "$(item.searches)\t$(item.branch_solves)\t$(item.ess)")
    end
    println()
    println("gradient audit mean cosine/relative error")
    for machines in setup.machines, method in methods
        item = gradient_summary["3x$(machines)"][string(method)]
        cosine = item["mean_cosine"]
        relative_error = item["mean_relative_error"]
        println("3x$(machines)\t$(method)\t$(cosine)\t$(relative_error)")
    end
    println("switching exposure lower/higher: ", switching_summary)
    println()
    println("[calibrated_time hardest 3x$(hardest_machines), horizon=$(hardest_horizon)]")
    println("method\tOOD loss\thigher-switch\tAUC\tseconds\tsteps")
    for method in methods
        item = summarize(
            rows, hardest_machines, hardest_horizon, :calibrated_time, method,
        )
        println("$(method)\t$(item.ood_loss)\t$(item.higher_switch_loss)\t" *
            "$(item.ood_auc)\t$(item.seconds)\t$(item.steps)")
    end
    println("hardest higher-switch SAGE-STE difference/95% CI: ",
        sage_ste.mean, " / [", sage_ste.lower, ", ", sage_ste.upper, "]")
    println("hardest higher-switch SAGE-hard difference/95% CI: ",
        sage_hard.mean, " / [", sage_hard.lower, ", ", sage_hard.upper, "]")
    println("crossover status: ", crossover_status)
    println("maximum logical violation: ", maximum_violation)
    for (name, passed) in (
        ("safety", safety_gate), ("structure", structure_ok),
        ("completion", completion_gate), ("stratification", stratification_ok),
        ("gradient audit", gradient_gate), ("estimator", estimator_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("switching audit CSV: ", audit_path)
    println("gradient audit CSV: ", gradient_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE13_DIFFICULTY_MAP_SMOKE=COMPLETE" :
        "PHASE13_DIFFICULTY_MAP=" * (phase13_pass ? "PASS" : "FAIL"))
    profile == :pilot && println("PHASE13_CROSSOVER=", crossover_status)
    profile == :pilot && !phase13_pass && exit(1)
end

main()
