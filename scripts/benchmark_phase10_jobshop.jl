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

function parse_seeds(default)
    values = parse.(Int, strip.(split(get(ENV, "SAGE_PHASE10_SEEDS", default), ',')))
    length(unique(values)) == length(values) || error("Phase 10 seeds must be unique.")
    return values
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
    finite_ess = filter(isfinite, [row.mean_effective_sample_size for row in selected])
    return (
        iid_task_loss = mean(row.iid_task_loss for row in selected),
        iid_makespan_gap = mean(row.iid_makespan_gap for row in selected),
        iid_csat = mean(row.iid_csat for row in selected),
        iid_order_agreement = mean(row.iid_order_agreement for row in selected),
        ood_task_loss = mean(row.ood_task_loss for row in selected),
        ood_makespan_gap = mean(row.ood_makespan_gap for row in selected),
        ood_csat = mean(row.ood_csat for row in selected),
        ood_order_agreement = mean(row.ood_order_agreement for row in selected),
        seconds = mean(row.training_seconds for row in selected),
        searches = sum(row.perturbed_global_searches for row in selected),
        complete_solves = sum(row.complete_branch_solves for row in selected),
        certificates = sum(row.sample_certificates for row in selected),
        switching = sum(row.switching_samples for row in selected),
        steps = sum(row.steps for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
    )
end

function profile_configuration(profile)
    if profile == :smoke
        return (
            seeds = parse_seeds("1"), train_size = 8, iid_size = 6, ood_size = 6,
            pretrain_epochs = 3, pretrain_batch = 4, fine_epochs = 1,
            max_steps = 3, samples = 2, radius = 0.50,
            learning_rate = 1e-3, evaluation_interval = 3,
        )
    elseif profile == :pilot
        seeds = parse_seeds("1,2,3,4,5")
        length(seeds) >= 5 || error("Phase 10 pilot requires at least five seeds.")
        return (
            seeds = seeds,
            train_size = parse(Int, get(ENV, "SAGE_PHASE10_TRAIN_SIZE", "16")),
            iid_size = parse(Int, get(ENV, "SAGE_PHASE10_IID_SIZE", "12")),
            ood_size = parse(Int, get(ENV, "SAGE_PHASE10_OOD_SIZE", "12")),
            pretrain_epochs = parse(Int, get(ENV, "SAGE_PHASE10_PRETRAIN_EPOCHS", "8")),
            pretrain_batch = parse(Int, get(ENV, "SAGE_PHASE10_PRETRAIN_BATCH", "8")),
            fine_epochs = parse(Int, get(ENV, "SAGE_PHASE10_FINE_EPOCHS", "1")),
            max_steps = parse(Int, get(ENV, "SAGE_PHASE10_MAX_STEPS", "6")),
            samples = parse(Int, get(ENV, "SAGE_PHASE10_SAMPLES", "4")),
            radius = parse(Float64, get(ENV, "SAGE_PHASE10_RADIUS", "0.50")),
            learning_rate = parse(Float64, get(ENV, "SAGE_PHASE10_LEARNING_RATE", "0.001")),
            evaluation_interval = parse(Int, get(ENV, "SAGE_PHASE10_EVAL_INTERVAL", "20")),
        )
    end
    throw(ArgumentError("SAGE_PHASE10_PROFILE must be smoke or pilot."))
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE10_PROFILE", "pilot")))
    setup = profile_configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE10_OUTPUT", "phase10_jobshop_results",
    ))
    mkpath(output_directory)
    methods = (:hard_local, :ste, :blackbox, :sage_cert, :sage_geometry)
    rows = NamedTuple[]
    trajectory_rows = NamedTuple[]
    config = JobShopConfig()
    maximum_oracle_violation = 0.0
    minimum_heuristic_gap = Inf
    mean_heuristic_gap = Float64[]

    warm = prepare_problem(999; train_size = 2, iid_size = 1, ood_size = 1)
    warm_network = pretrain_network(warm, 999; epochs = 1, batch_size = 2)
    for method in methods
        train_method(
            warm, warm_network, method, 999;
            max_steps = 1, samples = 2, evaluation_interval = 1,
        )
    end

    for seed in setup.seeds
        problem = prepare_problem(
            seed;
            config = config,
            train_size = setup.train_size,
            iid_size = setup.iid_size,
            ood_size = setup.ood_size,
        )
        for split in (problem.train, problem.iid, problem.ood)
            for instance in split.instances
                report = check_feasibility(instance.model, instance.oracle_starts)
                report.feasible || error("Phase 10 oracle feasibility audit failed.")
                maximum_oracle_violation = max(maximum_oracle_violation, report.max_violation)
                gap = (instance.heuristic_makespan - instance.oracle_makespan) /
                    instance.oracle_makespan
                minimum_heuristic_gap = min(minimum_heuristic_gap, gap)
                push!(mean_heuristic_gap, gap)
            end
        end

        pretrained = pretrain_network(
            problem,
            seed;
            epochs = setup.pretrain_epochs,
            batch_size = setup.pretrain_batch,
            learning_rate = 2e-3,
        )
        base_iid = evaluate_network(
            pretrained, problem.iid; constrained = false, config = config,
        )
        base_ood = evaluate_network(
            pretrained, problem.ood; constrained = false, config = config,
        )
        exact_iid = evaluate_network(pretrained, problem.iid; config = config)
        exact_ood = evaluate_network(pretrained, problem.ood; config = config)
        for (method, iid, ood) in (
            (:base, base_iid, base_ood),
            (:pretrained_exact, exact_iid, exact_ood),
        )
            push!(rows, (
                seed = seed, method = string(method),
                iid_task_loss = iid.task_loss, iid_makespan_gap = iid.makespan_gap,
                iid_csat = iid.csat, iid_order_agreement = iid.order_agreement,
                ood_task_loss = ood.task_loss, ood_makespan_gap = ood.makespan_gap,
                ood_csat = ood.csat, ood_order_agreement = ood.order_agreement,
                maximum_violation = max(iid.maximum_violation, ood.maximum_violation),
                unresolved = iid.unresolved + ood.unresolved,
                training_seconds = 0.0, steps = 0,
                perturbed_global_searches = 0, complete_branch_solves = 0,
                sample_certificates = 0, switching_samples = 0,
                unresolved_samples = 0, logical_failures = 0,
                mean_effective_sample_size = NaN,
            ))
        end

        shift = mod(seed - first(setup.seeds), length(methods))
        for method in circshift(collect(methods), -shift)
            result = train_method(
                problem,
                pretrained,
                method,
                seed;
                epochs = setup.fine_epochs,
                samples = setup.samples,
                radius = setup.radius,
                learning_rate = setup.learning_rate,
                evaluation_interval = setup.evaluation_interval,
                max_steps = setup.max_steps,
            )
            iid = evaluate_network(result.network, problem.iid; config = config)
            ood = evaluate_network(result.network, problem.ood; config = config)
            push!(rows, (
                seed = seed, method = string(method),
                iid_task_loss = iid.task_loss, iid_makespan_gap = iid.makespan_gap,
                iid_csat = iid.csat, iid_order_agreement = iid.order_agreement,
                ood_task_loss = ood.task_loss, ood_makespan_gap = ood.makespan_gap,
                ood_csat = ood.csat, ood_order_agreement = ood.order_agreement,
                maximum_violation = max(iid.maximum_violation, ood.maximum_violation),
                unresolved = iid.unresolved + ood.unresolved,
                training_seconds = result.training_seconds, steps = result.steps,
                perturbed_global_searches = result.perturbed_global_searches,
                complete_branch_solves = result.complete_branch_solves,
                sample_certificates = result.sample_certificates,
                switching_samples = result.switching_samples,
                unresolved_samples = result.unresolved_samples,
                logical_failures = result.logical_failures,
                mean_effective_sample_size = result.mean_effective_sample_size,
            ))
            for item in result.trajectory
                push!(trajectory_rows, (
                    seed = seed, method = string(method), step = item.step,
                    iid_task_loss = item.iid_task_loss,
                    iid_makespan_gap = item.iid_makespan_gap,
                    ood_task_loss = item.ood_task_loss,
                    ood_makespan_gap = item.ood_makespan_gap,
                ))
            end
            println(
                "[phase10] completed seed=$(seed) method=$(method) " *
                "seconds=$(round(result.training_seconds; digits = 3))",
            )
            flush(stdout)
        end
    end

    run_path = joinpath(output_directory, "phase10_runs.csv")
    trajectory_path = joinpath(output_directory, "phase10_trajectories.csv")
    summary_path = joinpath(output_directory, "phase10_summary.json")
    write_csv(run_path, propertynames(first(rows)), rows)
    write_csv(trajectory_path, propertynames(first(trajectory_rows)), trajectory_rows)

    summaries = Dict(method => summarize(rows, method) for method in methods)
    base = summarize(rows, :base)
    initial = summarize(rows, :pretrained_exact)
    hard = summaries[:hard_local]
    ste = summaries[:ste]
    blackbox = summaries[:blackbox]
    certificate = summaries[:sage_cert]
    geometry = summaries[:sage_geometry]
    constrained_rows = filter(row -> row.method != "base", rows)
    safety_gate = maximum_oracle_violation <= 1e-6 && all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.unresolved_samples == 0 && row.logical_failures == 0 &&
        row.maximum_violation <= 1e-6 for row in constrained_rows
    )
    adaptation_gate = mean(mean_heuristic_gap) >= 0.10 &&
        initial.ood_order_agreement < 0.95
    switching_rate = blackbox.switching / max(blackbox.steps * setup.samples, 1)
    switching_gate = switching_rate >= 0.02
    learning_gate = geometry.ood_task_loss <= initial.ood_task_loss
    quality_gate = geometry.ood_task_loss <= 1.20 * blackbox.ood_task_loss &&
        certificate.ood_task_loss <= 1.20 * blackbox.ood_task_loss
    cert_reduction = 1.0 - certificate.searches / blackbox.searches
    geometry_reduction = 1.0 - geometry.searches / blackbox.searches
    compute_gate = cert_reduction >= 0.20 && geometry_reduction >= 0.10
    quality_or_compute_gate = geometry.ood_task_loss <= 0.90 * blackbox.ood_task_loss ||
        compute_gate
    ess_gate = geometry.ess >= 0.60 * setup.samples
    phase10_pass = profile == :pilot && safety_gate && adaptation_gate &&
        switching_gate && learning_gate && quality_gate && quality_or_compute_gate &&
        ess_gate

    geometry_blackbox = [
        row_for(rows, :sage_geometry, seed).ood_task_loss -
            row_for(rows, :blackbox, seed).ood_task_loss
        for seed in setup.seeds
    ]
    geometry_ste = [
        row_for(rows, :sage_geometry, seed).ood_task_loss -
            row_for(rows, :ste, seed).ood_task_loss
        for seed in setup.seeds
    ]
    geometry_hard = [
        row_for(rows, :sage_geometry, seed).ood_task_loss -
            row_for(rows, :hard_local, seed).ood_task_loss
        for seed in setup.seeds
    ]
    blackbox_interval = paired_interval(geometry_blackbox)
    ste_interval = paired_interval(geometry_ste)
    hard_interval = paired_interval(geometry_hard)

    summary = Dict{String, Any}(
        "phase" => "phase10_native_jobshop",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" : phase10_pass ? "PASS" : "FAIL",
        "config" => Dict(
            string(field) => getfield(config, field) for field in fieldnames(JobShopConfig)
        ),
        "seeds" => setup.seeds,
        "train_size" => setup.train_size,
        "iid_size" => setup.iid_size,
        "ood_size" => setup.ood_size,
        "steps" => min(setup.max_steps, setup.fine_epochs * setup.train_size),
        "samples" => setup.samples,
        "radius" => setup.radius,
        "mean_heuristic_makespan_gap" => mean(mean_heuristic_gap),
        "pretrained_exact_ood_task_loss" => initial.ood_task_loss,
        "hard_local_ood_task_loss" => hard.ood_task_loss,
        "ste_ood_task_loss" => ste.ood_task_loss,
        "blackbox_ood_task_loss" => blackbox.ood_task_loss,
        "sage_cert_ood_task_loss" => certificate.ood_task_loss,
        "sage_geometry_ood_task_loss" => geometry.ood_task_loss,
        "sage_geometry_blackbox_difference" => blackbox_interval.mean,
        "sage_geometry_blackbox_ci95" => [blackbox_interval.lower, blackbox_interval.upper],
        "sage_geometry_ste_difference" => ste_interval.mean,
        "sage_geometry_ste_ci95" => [ste_interval.lower, ste_interval.upper],
        "sage_geometry_hard_difference" => hard_interval.mean,
        "sage_geometry_hard_ci95" => [hard_interval.lower, hard_interval.upper],
        "blackbox_switching_rate" => switching_rate,
        "sage_cert_global_reduction" => cert_reduction,
        "sage_geometry_global_reduction" => geometry_reduction,
        "sage_geometry_mean_ess" => geometry.ess,
        "safety_gate" => safety_gate,
        "adaptation_gate" => adaptation_gate,
        "switching_gate" => switching_gate,
        "learning_gate" => learning_gate,
        "quality_gate" => quality_gate,
        "compute_gate" => compute_gate,
        "quality_or_compute_gate" => quality_or_compute_gate,
        "ess_gate" => ess_gate,
        "pass" => phase10_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 10 native-disjunction job-shop ", profile)
    println("Julia version: ", VERSION)
    println("Contract: exact schedule forward; Eq.(11)-(12), Eq.(17)-(21); matched q/rho/N")
    println("jobs/machines/disjunctions/branches: 3/3/9/512")
    println("seeds/train/IID/OOD/steps/N/rho: ", length(setup.seeds), "/",
        setup.train_size, "/", setup.iid_size, "/", setup.ood_size, "/",
        min(setup.max_steps, setup.fine_epochs * setup.train_size), "/",
        setup.samples, "/", setup.radius)
    println("mean/min serial-dispatch makespan gap: ", mean(mean_heuristic_gap), "/",
        minimum_heuristic_gap)
    println("maximum oracle violation: ", maximum_oracle_violation)
    println()
    println("method\tIID task loss\tOOD task loss\tOOD makespan gap\tOOD CSAT\tseconds\tglobal searches\tbranch solves")
    for method in (:base, :pretrained_exact, methods...)
        item = method in methods ? summaries[method] : method == :base ? base : initial
        println("$(method)\t$(item.iid_task_loss)\t$(item.ood_task_loss)\t$(item.ood_makespan_gap)\t$(item.ood_csat)\t$(item.seconds)\t$(item.searches)\t$(item.complete_solves)")
    end
    println()
    println("SAGE-Geometry - black-box OOD task loss/95% CI: ",
        blackbox_interval.mean, " / [", blackbox_interval.lower, ", ",
        blackbox_interval.upper, "]")
    println("SAGE-Geometry - STE OOD task loss/95% CI: ",
        ste_interval.mean, " / [", ste_interval.lower, ", ", ste_interval.upper, "]")
    println("SAGE-Geometry - hard-local OOD task loss/95% CI: ",
        hard_interval.mean, " / [", hard_interval.lower, ", ", hard_interval.upper, "]")
    println("black-box switching rate: ", switching_rate)
    println("certificate/geometry global-search reduction: ", cert_reduction, "/", geometry_reduction)
    println("geometry mean ESS: ", geometry.ess, "/", setup.samples)
    for (name, passed) in (
        ("safety", safety_gate), ("adaptation", adaptation_gate),
        ("switching", switching_gate), ("learning", learning_gate),
        ("matched-quality", quality_gate), ("compute", compute_gate),
        ("quality-or-compute", quality_or_compute_gate),
        ("geometry ESS", ess_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE10_JOBSHOP_SMOKE=COMPLETE" :
        "PHASE10_JOBSHOP_DIAGNOSTIC=" * (phase10_pass ? "PASS" : "FAIL"))
    allow_recorded_diagnostic = get(
        ENV, "SAGE_PHASE10_ALLOW_DIAGNOSTIC", "0",
    ) == "1"
    profile == :pilot && !phase10_pass && !allow_recorded_diagnostic && exit(1)
end

main()
