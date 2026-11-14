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
            seeds = [1], train_size = 8, iid_size = 6, ood_size = 6,
            pretrain_epochs = 3, pretrain_batch = 4, max_steps = 3,
            samples = 4, radius = 0.5, learning_rate = 1e-3,
        )
    elseif profile == :pilot
        return (
            seeds = collect(1:5), train_size = 16, iid_size = 12, ood_size = 12,
            pretrain_epochs = 8, pretrain_batch = 8, max_steps = 6,
            samples = 4, radius = 0.5, learning_rate = 1e-3,
        )
    end
    throw(ArgumentError("SAGE_PHASE11_PROFILE must be smoke or pilot."))
end

function paired_interval(values)
    length(values) >= 5 || return (mean = mean(values), lower = NaN, upper = NaN)
    center = mean(values)
    halfwidth = 2.776 * std(values; corrected = true) / sqrt(length(values))
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

row_for(rows, method, seed) = only(filter(
    row -> row.method == string(method) && row.seed == seed, rows,
))

function summarize(rows, method)
    selected = filter(row -> row.method == string(method), rows)
    finite_ess = filter(isfinite, [row.mean_ess for row in selected])
    finite_radii = filter(isfinite, [row.mean_local_radius for row in selected])
    return (
        iid_loss = mean(row.iid_loss for row in selected),
        ood_loss = mean(row.ood_loss for row in selected),
        ood_makespan_gap = mean(row.ood_makespan_gap for row in selected),
        ood_csat = mean(row.ood_csat for row in selected),
        seconds = mean(row.seconds for row in selected),
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
        steps = sum(row.steps for row in selected),
    )
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE11_PROFILE", "smoke")))
    setup = configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE11_OUTPUT", "phase11_multiscale_smoke_results",
    ))
    mkpath(output_directory)
    methods = (:blackbox, :sage_geometry, :sage_multiscale)
    rows = NamedTuple[]
    trajectories = NamedTuple[]
    maximum_violation = 0.0

    warm = prepare_problem(999; train_size = 2, iid_size = 1, ood_size = 1)
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
        initial_iid = evaluate_network(pretrained, problem.iid; config = problem.config)
        initial = evaluate_network(pretrained, problem.ood; config = problem.config)
        push!(rows, (
            seed = seed, method = "pretrained_exact", iid_loss = initial_iid.task_loss,
            iid_csat = initial_iid.csat,
            ood_loss = initial.task_loss, ood_makespan_gap = initial.makespan_gap,
            ood_csat = initial.csat, maximum_violation = initial.maximum_violation,
            seconds = 0.0, steps = 0, searches = 0, branch_solves = 0,
            certificates = 0, switching = 0, unresolved = initial.unresolved,
            logical_failures = 0, mean_ess = NaN, mean_local_radius = NaN,
            positive_local_calls = 0, base_samples = 0, local_samples = 0,
            focused_samples = 0,
        ))
        maximum_violation = max(maximum_violation, initial.maximum_violation)

        shift = mod(seed - first(setup.seeds), length(methods))
        for method in circshift(collect(methods), -shift)
            result = train_method(
                problem,
                pretrained,
                method,
                seed;
                max_steps = setup.max_steps,
                samples = setup.samples,
                radius = setup.radius,
                learning_rate = setup.learning_rate,
                evaluation_interval = setup.max_steps,
            )
            iid = evaluate_network(result.network, problem.iid; config = problem.config)
            ood = evaluate_network(result.network, problem.ood; config = problem.config)
            maximum_violation = max(
                maximum_violation, iid.maximum_violation, ood.maximum_violation,
            )
            push!(rows, (
                seed = seed, method = string(method), iid_loss = iid.task_loss,
                iid_csat = iid.csat,
                ood_loss = ood.task_loss, ood_makespan_gap = ood.makespan_gap,
                ood_csat = ood.csat,
                maximum_violation = max(iid.maximum_violation, ood.maximum_violation),
                seconds = result.training_seconds, steps = result.steps,
                searches = result.perturbed_global_searches,
                branch_solves = result.complete_branch_solves,
                certificates = result.sample_certificates,
                switching = result.switching_samples,
                unresolved = result.unresolved_samples + iid.unresolved + ood.unresolved,
                logical_failures = result.logical_failures,
                mean_ess = result.mean_effective_sample_size,
                mean_local_radius = result.mean_local_radius,
                positive_local_calls = result.positive_local_radius_calls,
                base_samples = result.base_proposal_samples,
                local_samples = result.local_proposal_samples,
                focused_samples = result.focused_proposal_samples,
            ))
            for item in result.trajectory
                push!(trajectories, (
                    seed = seed, method = string(method), step = item.step,
                    iid_task_loss = item.iid_task_loss,
                    ood_task_loss = item.ood_task_loss,
                ))
            end
            println(
                "[phase11] completed seed=$(seed) method=$(method) " *
                "seconds=$(round(result.training_seconds; digits = 3))",
            )
            flush(stdout)
        end
    end

    run_path = joinpath(output_directory, "phase11_runs.csv")
    trajectory_path = joinpath(output_directory, "phase11_trajectories.csv")
    summary_path = joinpath(output_directory, "phase11_summary.json")
    write_csv(run_path, propertynames(first(rows)), rows)
    write_csv(trajectory_path, propertynames(first(trajectories)), trajectories)

    initial = summarize(rows, :pretrained_exact)
    blackbox = summarize(rows, :blackbox)
    fixed = summarize(rows, :sage_geometry)
    multiscale = summarize(rows, :sage_multiscale)
    fixed_reduction = 1.0 - fixed.searches / max(blackbox.searches, 1)
    multiscale_reduction = 1.0 - multiscale.searches / max(blackbox.searches, 1)
    incremental_reduction = 1.0 - multiscale.searches / max(fixed.searches, 1)
    branch_ratio = multiscale.branch_solves / max(fixed.branch_solves, 1)
    positive_local_rate = multiscale.positive_local_calls / max(multiscale.steps, 1)
    trained_rows = filter(row -> row.method != "pretrained_exact", rows)
    safety_gate = maximum_violation <= 1e-6 && all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.logical_failures == 0 for row in trained_rows
    )
    fixed_compute_gate = fixed_reduction >= 0.10
    multiscale_search_gate = incremental_reduction >= 0.20
    branch_solve_gate = branch_ratio <= 1.05
    local_coverage_gate = positive_local_rate >= 0.50 && multiscale.local_samples > 0
    ess_gate = multiscale.ess >= 0.40 * setup.samples
    learning_gate = multiscale.ood_loss <= initial.ood_loss
    quality_gate = multiscale.ood_loss <= 1.10 * blackbox.ood_loss &&
        multiscale.ood_loss <= 1.10 * fixed.ood_loss
    phase11_pass = profile == :pilot && safety_gate && fixed_compute_gate &&
        multiscale_search_gate && branch_solve_gate && local_coverage_gate &&
        ess_gate && learning_gate && quality_gate
    multiscale_fixed_interval = paired_interval([
        row_for(rows, :sage_multiscale, seed).ood_loss -
            row_for(rows, :sage_geometry, seed).ood_loss
        for seed in setup.seeds
    ])
    multiscale_blackbox_interval = paired_interval([
        row_for(rows, :sage_multiscale, seed).ood_loss -
            row_for(rows, :blackbox, seed).ood_loss
        for seed in setup.seeds
    ])
    summary = Dict{String, Any}(
        "phase" => "phase11_certified_multiscale",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" :
            phase11_pass ? "PASS" : "FAIL",
        "seeds" => setup.seeds,
        "samples" => setup.samples,
        "radius" => setup.radius,
        "pretrained_ood_loss" => initial.ood_loss,
        "blackbox_ood_loss" => blackbox.ood_loss,
        "fixed_geometry_ood_loss" => fixed.ood_loss,
        "multiscale_ood_loss" => multiscale.ood_loss,
        "fixed_global_reduction_vs_blackbox" => fixed_reduction,
        "multiscale_global_reduction_vs_blackbox" => multiscale_reduction,
        "multiscale_global_reduction_vs_fixed" => incremental_reduction,
        "multiscale_mean_ess" => multiscale.ess,
        "multiscale_mean_local_radius" => multiscale.local_radius,
        "multiscale_positive_local_calls" => multiscale.positive_local_calls,
        "multiscale_local_samples" => multiscale.local_samples,
        "multiscale_branch_solve_ratio_vs_fixed" => branch_ratio,
        "positive_local_call_rate" => positive_local_rate,
        "multiscale_fixed_difference" => multiscale_fixed_interval.mean,
        "multiscale_fixed_ci95" => [
            multiscale_fixed_interval.lower, multiscale_fixed_interval.upper,
        ],
        "multiscale_blackbox_difference" => multiscale_blackbox_interval.mean,
        "multiscale_blackbox_ci95" => [
            multiscale_blackbox_interval.lower,
            multiscale_blackbox_interval.upper,
        ],
        "maximum_violation" => maximum_violation,
        "safety_gate" => safety_gate,
        "fixed_compute_gate" => fixed_compute_gate,
        "multiscale_search_gate" => multiscale_search_gate,
        "branch_solve_gate" => branch_solve_gate,
        "local_coverage_gate" => local_coverage_gate,
        "ess_gate" => ess_gate,
        "learning_gate" => learning_gate,
        "quality_gate" => quality_gate,
        "pass" => phase11_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 11 certified multiscale ", profile)
    println("Julia version: ", VERSION)
    println("Target contract: fixed rho=0.5 compact q; exact q/p; Eq.(11)-(12),(17)-(21)")
    println("methods: black-box / strengthened fixed geometry / certified multiscale")
    println("method\tOOD loss\tOOD CSAT\tseconds\tglobal searches\tbranch solves\tESS")
    for (name, item) in (
        (:blackbox, blackbox), (:fixed_geometry, fixed),
        (:multiscale_geometry, multiscale),
    )
        println("$(name)\t$(item.ood_loss)\t$(item.ood_csat)\t$(item.seconds)\t" *
            "$(item.searches)\t$(item.branch_solves)\t$(item.ess)")
    end
    println("fixed/multiscale global reduction vs black-box: ",
        fixed_reduction, "/", multiscale_reduction)
    println("multiscale global reduction vs fixed: ", incremental_reduction)
    println("multiscale local radius/positive calls: ",
        multiscale.local_radius, "/", multiscale.positive_local_calls,
        " of ", multiscale.steps)
    println("multiscale base/local/focused samples: ", multiscale.base_samples, "/",
        multiscale.local_samples, "/", multiscale.focused_samples)
    println("multiscale branch-solve ratio vs fixed: ", branch_ratio)
    println("positive local-call rate: ", positive_local_rate)
    println("multiscale - fixed OOD loss/95% CI: ",
        multiscale_fixed_interval.mean, " / [", multiscale_fixed_interval.lower,
        ", ", multiscale_fixed_interval.upper, "]")
    println("multiscale - black-box OOD loss/95% CI: ",
        multiscale_blackbox_interval.mean, " / [",
        multiscale_blackbox_interval.lower, ", ",
        multiscale_blackbox_interval.upper, "]")
    println("maximum logical violation: ", maximum_violation)
    for (name, passed) in (
        ("safety", safety_gate),
        ("strengthened fixed compute", fixed_compute_gate),
        ("multiscale search", multiscale_search_gate),
        ("branch solves", branch_solve_gate),
        ("local coverage", local_coverage_gate),
        ("ESS", ess_gate),
        ("learning", learning_gate),
        ("matched quality", quality_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE11_MULTISCALE_SMOKE=COMPLETE" :
        "PHASE11_CERTIFIED_MULTISCALE=" * (phase11_pass ? "PASS" : "FAIL"))
    profile == :pilot && !phase11_pass && exit(1)
end

main()
