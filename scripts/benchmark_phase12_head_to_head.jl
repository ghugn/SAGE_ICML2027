using DisjunctiveNet
using Random
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
            seeds = [1], train_size = 8, iid_size = 6, ood_size = 6,
            pretrain_epochs = 3, pretrain_batch = 4, max_steps = 2,
            samples = 4, radius = 0.5, learning_rate = 1e-3,
            audit_samples = 4, wall_step_cap = 3,
        )
    elseif profile == :pilot
        return (
            seeds = collect(1:5), train_size = 16, iid_size = 12, ood_size = 12,
            pretrain_epochs = 8, pretrain_batch = 8, max_steps = 6,
            samples = 4, radius = 0.5, learning_rate = 1e-3,
            audit_samples = 8, wall_step_cap = 5,
        )
    end
    throw(ArgumentError("SAGE_PHASE12_PROFILE must be smoke or pilot."))
end

function paired_interval(values)
    length(values) >= 5 || return (mean = mean(values), lower = NaN, upper = NaN)
    center = mean(values)
    halfwidth = 2.776 * std(values; corrected = true) / sqrt(length(values))
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function subset_split(split::JobShopSplit, indices)
    members = Int.(collect(indices))
    isempty(members) && throw(ArgumentError("Phase 12 strata must be nonempty."))
    return JobShopSplit(
        split.raw_inputs[:, members], split.inputs[:, members],
        split.targets[:, members], split.instances[members],
    )
end

function evaluate_with_strata(network, problem, audit)
    iid = evaluate_network(network, problem.iid; config = problem.config)
    ood = evaluate_network(network, problem.ood; config = problem.config)
    lower = evaluate_network(
        network, subset_split(problem.ood, audit.lower); config = problem.config,
    )
    higher = evaluate_network(
        network, subset_split(problem.ood, audit.higher); config = problem.config,
    )
    return (iid = iid, ood = ood, lower = lower, higher = higher)
end

row_for(rows, protocol, method, seed) = only(filter(
    row -> row.protocol == string(protocol) && row.method == string(method) &&
        row.seed == seed,
    rows,
))

function summarize(rows, protocol, method)
    selected = filter(
        row -> row.protocol == string(protocol) && row.method == string(method), rows,
    )
    isempty(selected) && error("No Phase 12 rows for $(protocol)/$(method).")
    finite_ess = filter(isfinite, [row.mean_ess for row in selected])
    finite_radii = filter(isfinite, [row.mean_local_radius for row in selected])
    return (
        iid_loss = mean(row.iid_loss for row in selected),
        ood_loss = mean(row.ood_loss for row in selected),
        ood_makespan_gap = mean(row.ood_makespan_gap for row in selected),
        ood_csat = mean(row.ood_csat for row in selected),
        lower_switch_loss = mean(row.lower_switch_loss for row in selected),
        higher_switch_loss = mean(row.higher_switch_loss for row in selected),
        seconds = mean(row.seconds for row in selected),
        steps = sum(row.steps for row in selected),
        searches = sum(row.searches for row in selected),
        branch_solves = sum(row.branch_solves for row in selected),
        certificates = sum(row.certificates for row in selected),
        switching = sum(row.switching for row in selected),
        unresolved = sum(row.unresolved for row in selected),
        logical_failures = sum(row.logical_failures for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
        local_radius = isempty(finite_radii) ? NaN : mean(finite_radii),
        positive_local_calls = sum(row.positive_local_calls for row in selected),
        base_samples = sum(row.base_samples for row in selected),
        local_samples = sum(row.local_samples for row in selected),
        focused_samples = sum(row.focused_samples for row in selected),
    )
end

function result_row(seed, protocol, method, evaluation, result, planned_budget)
    return (
        seed = Int(seed), protocol = string(protocol), method = string(method),
        iid_loss = evaluation.iid.task_loss, iid_csat = evaluation.iid.csat,
        ood_loss = evaluation.ood.task_loss,
        ood_makespan_gap = evaluation.ood.makespan_gap,
        ood_csat = evaluation.ood.csat,
        lower_switch_loss = evaluation.lower.task_loss,
        higher_switch_loss = evaluation.higher.task_loss,
        maximum_violation = maximum((
            evaluation.iid.maximum_violation, evaluation.ood.maximum_violation,
            evaluation.lower.maximum_violation, evaluation.higher.maximum_violation,
        )),
        seconds = result.training_seconds, planned_budget_seconds = planned_budget,
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

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE12_PROFILE", "smoke")))
    setup = configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE12_OUTPUT", "phase12_head_to_head_smoke_results",
    ))
    mkpath(output_directory)
    methods = (:hard_local, :ste, :blackbox, :sage_geometry, :sage_multiscale)
    rows = NamedTuple[]
    trajectories = NamedTuple[]
    audit_rows = NamedTuple[]
    maximum_violation = 0.0
    lower_switch_rates = Float64[]
    higher_switch_rates = Float64[]
    audit_searches = 0

    # Warm every differentiated path before any timed comparison.
    warm = prepare_problem(999; train_size = 2, iid_size = 2, ood_size = 2)
    warm_network = pretrain_network(warm, 999; epochs = 1, batch_size = 2)
    for method in methods
        train_method(
            warm, warm_network, method, 999;
            max_steps = 1, samples = 2, radius = setup.radius,
            evaluation_interval = 1,
        )
    end

    for seed in setup.seeds
        problem = prepare_problem(
            seed;
            train_size = setup.train_size,
            iid_size = setup.iid_size,
            ood_size = setup.ood_size,
        )
        pretrained = pretrain_network(
            problem,
            seed;
            epochs = setup.pretrain_epochs,
            batch_size = setup.pretrain_batch,
            learning_rate = 2e-3,
        )
        audit = switching_exposure_partition(
            pretrained, problem.ood, seed;
            radius = setup.radius, samples = setup.audit_samples,
        )
        lower_set = Set(audit.lower)
        for index in eachindex(audit.scores)
            push!(audit_rows, (
                seed = seed, instance = index,
                switch_count = round(Int, audit.scores[index] * setup.audit_samples),
                probes = setup.audit_samples, switch_rate = audit.scores[index],
                stratum = index in lower_set ? "lower_switch" : "higher_switch",
            ))
        end
        push!(lower_switch_rates, audit.lower_rate)
        push!(higher_switch_rates, audit.higher_rate)
        audit_searches += audit.searches

        initial_eval = evaluate_with_strata(pretrained, problem, audit)
        maximum_violation = max(
            maximum_violation, initial_eval.iid.maximum_violation,
            initial_eval.ood.maximum_violation,
        )
        push!(rows, (
            seed = seed, protocol = "reference", method = "pretrained_exact",
            iid_loss = initial_eval.iid.task_loss, iid_csat = initial_eval.iid.csat,
            ood_loss = initial_eval.ood.task_loss,
            ood_makespan_gap = initial_eval.ood.makespan_gap,
            ood_csat = initial_eval.ood.csat,
            lower_switch_loss = initial_eval.lower.task_loss,
            higher_switch_loss = initial_eval.higher.task_loss,
            maximum_violation = max(
                initial_eval.iid.maximum_violation, initial_eval.ood.maximum_violation,
            ),
            seconds = 0.0, planned_budget_seconds = 0.0, steps = 0,
            searches = 0, branch_solves = 0, certificates = 0, switching = 0,
            unresolved = initial_eval.iid.unresolved + initial_eval.ood.unresolved,
            logical_failures = 0, mean_ess = NaN, mean_local_radius = NaN,
            positive_local_calls = 0, base_samples = 0, local_samples = 0,
            focused_samples = 0,
        ))

        primary_results = Dict{Symbol, Any}()
        shift = mod(seed - first(setup.seeds), length(methods))
        primary_order = circshift(collect(methods), -shift)
        for method in primary_order
            result = train_method(
                problem, pretrained, method, seed;
                max_steps = setup.max_steps, samples = setup.samples,
                radius = setup.radius, learning_rate = setup.learning_rate,
                evaluation_interval = setup.max_steps,
            )
            evaluation = evaluate_with_strata(result.network, problem, audit)
            row = result_row(
                seed, :matched_steps, method, evaluation, result, 0.0,
            )
            push!(rows, row)
            primary_results[method] = result
            maximum_violation = max(maximum_violation, row.maximum_violation)
            for item in result.trajectory
                push!(trajectories, (
                    seed = seed, protocol = "matched_steps", method = string(method),
                    step = item.step, iid_task_loss = item.iid_task_loss,
                    ood_task_loss = item.ood_task_loss,
                ))
            end
            println(
                "[phase12/matched_steps] seed=$(seed) method=$(method) " *
                "seconds=$(round(result.training_seconds; digits = 3))",
            )
            flush(stdout)
        end

        target_seconds = maximum(result.training_seconds for result in values(primary_results))
        planned_steps = Dict{Symbol, Int}()
        for method in methods
            result = primary_results[method]
            seconds_per_step = result.training_seconds / max(result.steps, 1)
            estimated = floor(Int, target_seconds / max(seconds_per_step, eps(Float64)))
            planned_steps[method] = clamp(
                estimated, setup.max_steps, setup.wall_step_cap * setup.max_steps,
            )
        end

        wall_order = reverse(primary_order)
        for method in wall_order
            steps = planned_steps[method]
            epochs = cld(steps, setup.train_size)
            result = train_method(
                problem, pretrained, method, seed;
                epochs = epochs, max_steps = steps, samples = setup.samples,
                radius = setup.radius, learning_rate = setup.learning_rate,
                evaluation_interval = steps,
            )
            evaluation = evaluate_with_strata(result.network, problem, audit)
            row = result_row(
                seed, :calibrated_time, method, evaluation, result, target_seconds,
            )
            push!(rows, row)
            maximum_violation = max(maximum_violation, row.maximum_violation)
            for item in result.trajectory
                push!(trajectories, (
                    seed = seed, protocol = "calibrated_time", method = string(method),
                    step = item.step, iid_task_loss = item.iid_task_loss,
                    ood_task_loss = item.ood_task_loss,
                ))
            end
            println(
                "[phase12/calibrated_time] seed=$(seed) method=$(method) " *
                "steps=$(steps) seconds=$(round(result.training_seconds; digits = 3))",
            )
            flush(stdout)
        end
    end

    run_path = joinpath(output_directory, "phase12_runs.csv")
    trajectory_path = joinpath(output_directory, "phase12_trajectories.csv")
    audit_path = joinpath(output_directory, "phase12_switching_audit.csv")
    summary_path = joinpath(output_directory, "phase12_summary.json")
    write_csv(run_path, propertynames(first(rows)), rows)
    write_csv(trajectory_path, propertynames(first(trajectories)), trajectories)
    write_csv(audit_path, propertynames(first(audit_rows)), audit_rows)

    initial = summarize(rows, :reference, :pretrained_exact)
    matched = Dict(method => summarize(rows, :matched_steps, method) for method in methods)
    timed = Dict(method => summarize(rows, :calibrated_time, method) for method in methods)
    hard = matched[:hard_local]
    ste = matched[:ste]
    blackbox = matched[:blackbox]
    fixed = matched[:sage_geometry]
    multiscale = matched[:sage_multiscale]

    safety_gate = maximum_violation <= 1e-6 && all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.logical_failures == 0 && row.maximum_violation <= 1e-6
        for row in rows
    )
    stratification_gate = all(
        high >= low for (low, high) in zip(lower_switch_rates, higher_switch_rates)
    ) && length(audit_rows) == length(setup.seeds) * setup.ood_size
    learning_gate = multiscale.ood_loss <= initial.ood_loss
    heuristic_noninferiority_gate = multiscale.ood_loss <= 1.10 * ste.ood_loss &&
        multiscale.ood_loss <= 1.10 * hard.ood_loss
    search_reduction_blackbox = 1.0 - multiscale.searches / max(blackbox.searches, 1)
    search_reduction_fixed = 1.0 - multiscale.searches / max(fixed.searches, 1)
    search_gate = search_reduction_blackbox >= 0.20 && search_reduction_fixed >= 0.10
    branch_ratio = multiscale.branch_solves / max(fixed.branch_solves, 1)
    branch_gate = branch_ratio <= 1.05
    positive_local_rate = multiscale.positive_local_calls / max(multiscale.steps, 1)
    estimator_gate = positive_local_rate >= 0.50 &&
        multiscale.local_samples > 0 && multiscale.ess >= 0.40 * setup.samples
    higher_switch_gate = multiscale.higher_switch_loss <=
        1.10 * ste.higher_switch_loss && multiscale.higher_switch_loss <=
        1.10 * hard.higher_switch_loss
    phase12_pass = profile == :pilot && safety_gate && stratification_gate &&
        learning_gate && heuristic_noninferiority_gate && search_gate && branch_gate &&
        estimator_gate && higher_switch_gate

    multi_ste = paired_interval([
        row_for(rows, :matched_steps, :sage_multiscale, seed).ood_loss -
            row_for(rows, :matched_steps, :ste, seed).ood_loss
        for seed in setup.seeds
    ])
    multi_hard = paired_interval([
        row_for(rows, :matched_steps, :sage_multiscale, seed).ood_loss -
            row_for(rows, :matched_steps, :hard_local, seed).ood_loss
        for seed in setup.seeds
    ])
    multi_ste_higher = paired_interval([
        row_for(rows, :matched_steps, :sage_multiscale, seed).higher_switch_loss -
            row_for(rows, :matched_steps, :ste, seed).higher_switch_loss
        for seed in setup.seeds
    ])
    multi_hard_higher = paired_interval([
        row_for(rows, :matched_steps, :sage_multiscale, seed).higher_switch_loss -
            row_for(rows, :matched_steps, :hard_local, seed).higher_switch_loss
        for seed in setup.seeds
    ])

    method_summary(item) = Dict(
        "ood_loss" => item.ood_loss,
        "higher_switch_loss" => item.higher_switch_loss,
        "lower_switch_loss" => item.lower_switch_loss,
        "seconds" => item.seconds,
        "steps" => item.steps,
        "global_searches" => item.searches,
        "branch_solves" => item.branch_solves,
        "ess" => item.ess,
    )
    summary = Dict{String, Any}(
        "phase" => "phase12_head_to_head",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" :
            phase12_pass ? "PASS" : "FAIL",
        "seeds" => setup.seeds,
        "samples" => setup.samples,
        "radius" => setup.radius,
        "matched_steps" => Dict(
            string(method) => method_summary(matched[method]) for method in methods
        ),
        "calibrated_time" => Dict(
            string(method) => method_summary(timed[method]) for method in methods
        ),
        "pretrained_ood_loss" => initial.ood_loss,
        "lower_switch_probe_rate" => mean(lower_switch_rates),
        "higher_switch_probe_rate" => mean(higher_switch_rates),
        "switching_audit_global_searches" => audit_searches,
        "multiscale_ste_difference" => multi_ste.mean,
        "multiscale_ste_ci95" => [multi_ste.lower, multi_ste.upper],
        "multiscale_hard_difference" => multi_hard.mean,
        "multiscale_hard_ci95" => [multi_hard.lower, multi_hard.upper],
        "multiscale_ste_higher_switch_difference" => multi_ste_higher.mean,
        "multiscale_ste_higher_switch_ci95" => [
            multi_ste_higher.lower, multi_ste_higher.upper,
        ],
        "multiscale_hard_higher_switch_difference" => multi_hard_higher.mean,
        "multiscale_hard_higher_switch_ci95" => [
            multi_hard_higher.lower, multi_hard_higher.upper,
        ],
        "multiscale_global_reduction_vs_blackbox" => search_reduction_blackbox,
        "multiscale_global_reduction_vs_fixed" => search_reduction_fixed,
        "multiscale_branch_solve_ratio_vs_fixed" => branch_ratio,
        "multiscale_positive_local_call_rate" => positive_local_rate,
        "maximum_violation" => maximum_violation,
        "safety_gate" => safety_gate,
        "stratification_gate" => stratification_gate,
        "learning_gate" => learning_gate,
        "heuristic_noninferiority_gate" => heuristic_noninferiority_gate,
        "search_gate" => search_gate,
        "branch_gate" => branch_gate,
        "estimator_gate" => estimator_gate,
        "higher_switch_gate" => higher_switch_gate,
        "pass" => phase12_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 12 matched head-to-head ", profile)
    println("Julia version: ", VERSION)
    println("Contract: identical exact forwards/initialization/data; fixed rho=0.5; Eq.(11)-(12),(17)-(21)")
    println("methods: hard-local / STE / black-box / fixed geometry / certified multiscale")
    println("switching audit lower/higher mean probe rate: ",
        mean(lower_switch_rates), "/", mean(higher_switch_rates))
    println()
    for protocol in (:matched_steps, :calibrated_time)
        collection = protocol == :matched_steps ? matched : timed
        println("[", protocol, "]")
        println("method\tOOD loss\tlower-switch\thigher-switch\tseconds\tsteps\tglobal searches\tbranch solves\tESS")
        for method in methods
            item = collection[method]
            println("$(method)\t$(item.ood_loss)\t$(item.lower_switch_loss)\t" *
                "$(item.higher_switch_loss)\t$(item.seconds)\t$(item.steps)\t" *
                "$(item.searches)\t$(item.branch_solves)\t$(item.ess)")
        end
        println()
    end
    println("multiscale - STE OOD loss/95% CI: ", multi_ste.mean, " / [",
        multi_ste.lower, ", ", multi_ste.upper, "]")
    println("multiscale - hard-local OOD loss/95% CI: ", multi_hard.mean, " / [",
        multi_hard.lower, ", ", multi_hard.upper, "]")
    println("higher-switch multiscale - STE loss/95% CI: ",
        multi_ste_higher.mean, " / [", multi_ste_higher.lower, ", ",
        multi_ste_higher.upper, "]")
    println("higher-switch multiscale - hard-local loss/95% CI: ",
        multi_hard_higher.mean, " / [", multi_hard_higher.lower, ", ",
        multi_hard_higher.upper, "]")
    println("multiscale global reduction vs black-box/fixed: ",
        search_reduction_blackbox, "/", search_reduction_fixed)
    println("multiscale branch-solve ratio vs fixed: ", branch_ratio)
    println("multiscale positive-local rate/ESS: ", positive_local_rate, "/",
        multiscale.ess, " of ", setup.samples)
    println("maximum logical violation: ", maximum_violation)
    for (name, passed) in (
        ("safety", safety_gate), ("switching stratification", stratification_gate),
        ("learning", learning_gate),
        ("heuristic non-inferiority", heuristic_noninferiority_gate),
        ("global-search", search_gate), ("branch-solve", branch_gate),
        ("estimator", estimator_gate), ("higher-switch non-inferiority", higher_switch_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("switching audit CSV: ", audit_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE12_HEAD_TO_HEAD_SMOKE=COMPLETE" :
        "PHASE12_HEAD_TO_HEAD=" * (phase12_pass ? "PASS" : "FAIL"))
    profile == :pilot && !phase12_pass && exit(1)
end

main()
