using DisjunctiveNet
using Random
using Statistics

include(joinpath(@__DIR__, "phase9_cooling_common.jl"))
using .Phase9Cooling

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
    seeds = parse.(Int, strip.(split(get(ENV, "SAGE_PHASE9_SEEDS", default), ',')))
    length(unique(seeds)) == length(seeds) || error("Phase 9 seeds must be unique.")
    return seeds
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
        iid_mse = mean(row.iid_mse for row in selected),
        iid_csat = mean(row.iid_csat for row in selected),
        ood_mse = mean(row.ood_mse for row in selected),
        ood_csat = mean(row.ood_csat for row in selected),
        ood_branch_agreement = mean(row.ood_branch_agreement for row in selected),
        seconds = mean(row.training_seconds for row in selected),
        global_searches = sum(row.perturbed_global_searches for row in selected),
        complete_branch_solves = sum(row.complete_branch_solves for row in selected),
        certificates = sum(row.sample_certificates for row in selected),
        geometry_solves = sum(row.geometry_branch_solves for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
    )
end

function profile_configuration(profile)
    if profile == :smoke
        return (
            seeds = parse_seeds("1"), train_size = 40, iid_size = 40, ood_size = 40,
            pretrain_epochs = 2, pretrain_batch = 20, fine_epochs = 1,
            max_steps = 20, samples = 4, radius = 0.15,
            learning_rate = 1e-3, evaluation_interval = 20,
        )
    elseif profile == :pilot
        seeds = parse_seeds("1,2,3,4,5")
        length(seeds) >= 5 || error("Phase 9 pilot requires at least five seeds.")
        return (
            seeds = seeds,
            train_size = parse(Int, get(ENV, "SAGE_PHASE9_TRAIN_SIZE", "250")),
            iid_size = parse(Int, get(ENV, "SAGE_PHASE9_IID_SIZE", "500")),
            ood_size = parse(Int, get(ENV, "SAGE_PHASE9_OOD_SIZE", "500")),
            pretrain_epochs = parse(Int, get(ENV, "SAGE_PHASE9_PRETRAIN_EPOCHS", "5")),
            pretrain_batch = parse(Int, get(ENV, "SAGE_PHASE9_PRETRAIN_BATCH", "64")),
            fine_epochs = parse(Int, get(ENV, "SAGE_PHASE9_FINE_EPOCHS", "1")),
            max_steps = parse(Int, get(ENV, "SAGE_PHASE9_MAX_STEPS", "250")),
            samples = parse(Int, get(ENV, "SAGE_PHASE9_SAMPLES", "8")),
            radius = parse(Float64, get(ENV, "SAGE_PHASE9_RADIUS", "0.15")),
            learning_rate = parse(Float64, get(ENV, "SAGE_PHASE9_LEARNING_RATE", "0.001")),
            evaluation_interval = parse(Int, get(ENV, "SAGE_PHASE9_EVAL_INTERVAL", "250")),
        )
    end
    throw(ArgumentError("SAGE_PHASE9_PROFILE must be smoke or pilot."))
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE9_PROFILE", "pilot")))
    setup = profile_configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE9_OUTPUT", "phase9_cooling_results",
    ))
    mkpath(output_directory)
    methods = (:hard_local, :ste, :blackbox, :sage_cert, :sage_geometry)
    rows = NamedTuple[]
    trajectory_rows = NamedTuple[]
    reconstruction = CoolingConfig()
    maximum_oracle_violation = 0.0
    maximum_active_rules = 0
    mean_ood_active_rules = Float64[]

    # Compile all paths on a separate tiny task before recorded timing.
    warm = prepare_problem(999; config = reconstruction, train_size = 8, iid_size = 4, ood_size = 4)
    warm_network = pretrain_network(warm, 999; epochs = 1, batch_size = 4)
    for method in methods
        train_method(
            warm, warm_network, method, 999;
            max_steps = 1, samples = 2, evaluation_interval = 1,
        )
    end

    for seed in setup.seeds
        problem = prepare_problem(
            seed;
            config = reconstruction,
            train_size = setup.train_size,
            iid_size = setup.iid_size,
            ood_size = setup.ood_size,
        )
        maximum_active_rules = max(
            maximum_active_rules,
            maximum(length.(problem.train.active_rules)),
            maximum(length.(problem.iid.active_rules)),
            maximum(length.(problem.ood.active_rules)),
        )
        push!(mean_ood_active_rules, mean(length.(problem.ood.active_rules)))
        for split in (problem.train, problem.iid, problem.ood), index in eachindex(split.models)
            report = check_feasibility(split.models[index], split.oracle_outputs[:, index])
            report.feasible || error("Phase 9 oracle feasibility audit failed.")
            maximum_oracle_violation = max(maximum_oracle_violation, report.max_violation)
        end

        pretrained = pretrain_network(
            problem,
            seed;
            epochs = setup.pretrain_epochs,
            batch_size = setup.pretrain_batch,
            learning_rate = setup.learning_rate,
        )
        base_iid = evaluate_network(pretrained, problem.iid; constrained = false)
        base_ood = evaluate_network(pretrained, problem.ood; constrained = false)
        exact_iid = evaluate_network(pretrained, problem.iid; constrained = true)
        exact_ood = evaluate_network(pretrained, problem.ood; constrained = true)
        for (method, iid_metrics, ood_metrics) in (
            (:base, base_iid, base_ood),
            (:pretrained_exact, exact_iid, exact_ood),
        )
            push!(rows, (
                seed = seed, method = string(method), iid_mse = iid_metrics.mse,
                iid_csat = iid_metrics.csat, iid_branch_agreement = iid_metrics.branch_agreement,
                ood_mse = ood_metrics.mse, ood_csat = ood_metrics.csat,
                ood_branch_agreement = ood_metrics.branch_agreement,
                maximum_violation = max(iid_metrics.maximum_violation, ood_metrics.maximum_violation),
                unresolved = iid_metrics.unresolved + ood_metrics.unresolved,
                training_seconds = 0.0, center_global_searches = 0,
                center_leaf_solves = 0, center_relaxation_solves = 0,
                perturbed_global_searches = 0, reference_branch_solves = 0,
                competitor_branch_solves = 0, geometry_branch_solves = 0,
                sample_certificates = 0, lazy_leaf_solves = 0,
                lazy_relaxation_solves = 0, switching_samples = 0,
                unresolved_samples = 0, logical_failures = 0,
                complete_branch_solves = 0, mean_effective_sample_size = NaN,
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
            iid_metrics = evaluate_network(result.network, problem.iid; constrained = true)
            ood_metrics = evaluate_network(result.network, problem.ood; constrained = true)
            complete_branch_solves = result.center_leaf_solves +
                result.reference_branch_solves + result.competitor_branch_solves +
                result.geometry_branch_solves + result.lazy_leaf_solves
            push!(rows, (
                seed = seed, method = string(method), iid_mse = iid_metrics.mse,
                iid_csat = iid_metrics.csat, iid_branch_agreement = iid_metrics.branch_agreement,
                ood_mse = ood_metrics.mse, ood_csat = ood_metrics.csat,
                ood_branch_agreement = ood_metrics.branch_agreement,
                maximum_violation = max(iid_metrics.maximum_violation, ood_metrics.maximum_violation),
                unresolved = iid_metrics.unresolved + ood_metrics.unresolved,
                training_seconds = result.training_seconds,
                center_global_searches = result.center_global_searches,
                center_leaf_solves = result.center_leaf_solves,
                center_relaxation_solves = result.center_relaxation_solves,
                perturbed_global_searches = result.perturbed_global_searches,
                reference_branch_solves = result.reference_branch_solves,
                competitor_branch_solves = result.competitor_branch_solves,
                geometry_branch_solves = result.geometry_branch_solves,
                sample_certificates = result.sample_certificates,
                lazy_leaf_solves = result.lazy_leaf_solves,
                lazy_relaxation_solves = result.lazy_relaxation_solves,
                switching_samples = result.switching_samples,
                unresolved_samples = result.unresolved_samples,
                logical_failures = result.logical_failures,
                complete_branch_solves = complete_branch_solves,
                mean_effective_sample_size = result.mean_effective_sample_size,
            ))
            for item in result.trajectory
                push!(trajectory_rows, (
                    seed = seed, method = string(method), step = item.step,
                    iid_mse = item.iid_mse, iid_csat = item.iid_csat,
                    ood_mse = item.ood_mse, ood_csat = item.ood_csat,
                ))
            end
            println(
                "[phase9] completed seed=$(seed) method=$(method) " *
                "seconds=$(round(result.training_seconds; digits = 3))",
            )
            flush(stdout)
        end
    end

    columns = propertynames(first(rows))
    trajectory_columns = propertynames(first(trajectory_rows))
    run_path = joinpath(output_directory, "phase9_runs.csv")
    trajectory_path = joinpath(output_directory, "phase9_trajectories.csv")
    summary_path = joinpath(output_directory, "phase9_summary.json")
    write_csv(run_path, columns, rows)
    write_csv(trajectory_path, trajectory_columns, trajectory_rows)

    summaries = Dict(method => summarize(rows, method) for method in methods)
    base = summarize(rows, :base)
    initial = summarize(rows, :pretrained_exact)
    blackbox = summaries[:blackbox]
    certificate = summaries[:sage_cert]
    geometry = summaries[:sage_geometry]
    constrained_rows = filter(row -> row.method != "base", rows)
    safety_gate = maximum_oracle_violation <= 1e-6 && all(
        row.iid_csat == 1.0 && row.ood_csat == 1.0 && row.unresolved == 0 &&
        row.unresolved_samples == 0 && row.logical_failures == 0 &&
        row.maximum_violation <= 1e-6 for row in constrained_rows
    )
    native_gate = maximum_active_rules >= 3 && mean(mean_ood_active_rules) >= 1.5
    learning_gate = geometry.ood_mse <= initial.ood_mse
    matched_quality_gate = geometry.ood_mse <= 1.15 * blackbox.ood_mse &&
        certificate.ood_mse <= 1.15 * blackbox.ood_mse
    cert_reduction = 1.0 - certificate.global_searches / blackbox.global_searches
    geometry_reduction = 1.0 - geometry.global_searches / blackbox.global_searches
    compute_gate = cert_reduction >= 0.25 && geometry_reduction >= 0.15
    geometry_ess_gate = geometry.ess >= 0.70 * setup.samples
    phase9_pass = profile == :pilot && safety_gate && native_gate && learning_gate &&
        matched_quality_gate && compute_gate && geometry_ess_gate

    geometry_blackbox = [
        row_for(rows, :sage_geometry, seed).ood_mse - row_for(rows, :blackbox, seed).ood_mse
        for seed in setup.seeds
    ]
    interval = paired_interval(geometry_blackbox)
    summary = Dict{String, Any}(
        "phase" => "phase9_cooling_paper_spec_reconstruction",
        "profile" => string(profile),
        "upstream_cooling_experiment_code_available" => false,
        "reconstruction_config" => Dict(
            string(field) => getfield(reconstruction, field)
            for field in fieldnames(CoolingConfig)
        ),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" : phase9_pass ? "PASS" : "FAIL",
        "seeds" => setup.seeds,
        "train_size" => setup.train_size,
        "iid_size" => setup.iid_size,
        "ood_size" => setup.ood_size,
        "pretrain_epochs" => setup.pretrain_epochs,
        "fine_epochs" => setup.fine_epochs,
        "fine_steps" => min(setup.max_steps, setup.fine_epochs * setup.train_size),
        "samples" => setup.samples,
        "radius" => setup.radius,
        "maximum_active_rules" => maximum_active_rules,
        "mean_ood_active_rules" => mean(mean_ood_active_rules),
        "maximum_oracle_violation" => maximum_oracle_violation,
        "base_iid_mse" => base.iid_mse,
        "base_ood_mse" => base.ood_mse,
        "pretrained_exact_ood_mse" => initial.ood_mse,
        "blackbox_ood_mse" => blackbox.ood_mse,
        "sage_cert_ood_mse" => certificate.ood_mse,
        "sage_geometry_ood_mse" => geometry.ood_mse,
        "sage_geometry_blackbox_difference" => interval.mean,
        "sage_geometry_blackbox_ci95" => [interval.lower, interval.upper],
        "blackbox_global_searches" => blackbox.global_searches,
        "sage_cert_global_searches" => certificate.global_searches,
        "sage_geometry_global_searches" => geometry.global_searches,
        "sage_cert_global_reduction" => cert_reduction,
        "sage_geometry_global_reduction" => geometry_reduction,
        "sage_geometry_mean_ess" => geometry.ess,
        "safety_gate" => safety_gate,
        "native_disjunction_gate" => native_gate,
        "learning_gate" => learning_gate,
        "matched_quality_gate" => matched_quality_gate,
        "compute_gate" => compute_gate,
        "geometry_ess_gate" => geometry_ess_gate,
        "pass" => phase9_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 9 native-disjunction cooling-control ", profile)
    println("Julia version: ", VERSION)
    println("Source status: paper-spec reconstruction; unpublished numeric constants declared in CoolingConfig")
    println("CoolingConfig: ", join(
        ("$(field)=$(getfield(reconstruction, field))" for field in fieldnames(CoolingConfig)),
        ", ",
    ))
    println("seeds/train/IID/OOD/steps/N/rho: ", length(setup.seeds), "/", setup.train_size,
        "/", setup.iid_size, "/", setup.ood_size, "/", min(setup.max_steps, setup.fine_epochs * setup.train_size),
        "/", setup.samples, "/", setup.radius)
    println("maximum/mean-OOD active rules: ", maximum_active_rules, "/", mean(mean_ood_active_rules))
    println("maximum oracle violation: ", maximum_oracle_violation)
    println()
    println("method\tIID MSE\tIID CSAT\tOOD MSE\tOOD CSAT\tseconds\tglobal searches\tcomplete branch solves")
    for method in (:base, :pretrained_exact, methods...)
        item = method in methods ? summaries[method] : method == :base ? base : initial
        println("$(method)\t$(item.iid_mse)\t$(item.iid_csat)\t$(item.ood_mse)\t$(item.ood_csat)\t$(item.seconds)\t$(item.global_searches)\t$(item.complete_branch_solves)")
    end
    println()
    println("SAGE-Geometry - black-box OOD MSE/95% CI: ", interval.mean,
        " / [", interval.lower, ", ", interval.upper, "]")
    println("certificate/geometry global-search reduction: ", cert_reduction, "/", geometry_reduction)
    println("geometry mean ESS: ", geometry.ess, "/", setup.samples)
    println("safety gate: ", safety_gate ? "PASS" : "FAIL")
    println("native-disjunction gate: ", native_gate ? "PASS" : "FAIL")
    println("learning gate: ", learning_gate ? "PASS" : "FAIL")
    println("matched-quality gate: ", matched_quality_gate ? "PASS" : "FAIL")
    println("compute gate: ", compute_gate ? "PASS" : "FAIL")
    println("geometry ESS gate: ", geometry_ess_gate ? "PASS" : "FAIL")
    println("run CSV: ", run_path)
    println("trajectory CSV: ", trajectory_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE9_COOLING_SMOKE=COMPLETE" :
        "PHASE9_COOLING_PILOT=" * (phase9_pass ? "PASS" : "FAIL"))
    profile == :pilot && !phase9_pass && exit(1)
end

main()
