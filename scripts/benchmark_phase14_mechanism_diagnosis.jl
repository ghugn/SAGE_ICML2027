using DisjunctiveNet
using LinearAlgebra
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
            seeds = [1], rhos = [0.30, 0.50], sample_counts = [4, 8],
            train_size = 4, iid_size = 2, ood_size = 2,
            pretrain_epochs = 2, pretrain_batch = 2,
            reference_samples = 4, evaluation_samples = 4,
            step_size = 0.03,
        )
    elseif profile == :pilot
        return (
            seeds = collect(1:5), rhos = [0.10, 0.30, 0.50],
            sample_counts = [4, 8, 16, 32],
            train_size = 16, iid_size = 12, ood_size = 12,
            pretrain_epochs = 8, pretrain_batch = 8,
            reference_samples = 64, evaluation_samples = 16,
            step_size = 0.03,
        )
    end
    throw(ArgumentError("SAGE_PHASE14_PROFILE must be smoke or pilot."))
end

function exact_task_loss(instance, point, config)
    result = lazy_branch_search(instance.model, point; store_prune_records = false)
    result.status == :optimal || error("Phase 14 exact projection unresolved.")
    result.report !== nothing && result.report.feasible || error(
        "Phase 14 exact projection is not logically feasible.",
    )
    return (
        loss = Phase10JobShop._task_loss(result.y, instance, config),
        choices = result.choices,
        violation = result.report.max_violation,
        leaf_solves = result.leaf_solves,
        relaxation_solves = result.relaxation_solves,
    )
end

function smoothed_task_loss(instance, center, perturbations, radius, config)
    total = 0.0
    maximum_violation = 0.0
    leaf_solves = 0
    relaxation_solves = 0
    for column in axes(perturbations, 2)
        result = lazy_branch_search(
            instance.model,
            center .+ Float64(radius) .* perturbations[:, column];
            store_prune_records = false,
        )
        result.status == :optimal || error("Phase 14 smoothed evaluation unresolved.")
        result.report !== nothing && result.report.feasible || error(
            "Phase 14 smoothed evaluation is not logically feasible.",
        )
        total += Phase10JobShop._task_loss(result.y, instance, config)
        maximum_violation = max(maximum_violation, result.report.max_violation)
        leaf_solves += result.leaf_solves
        relaxation_solves += result.relaxation_solves
    end
    return (
        loss = total / size(perturbations, 2),
        violation = maximum_violation,
        searches = size(perturbations, 2),
        leaf_solves = leaf_solves,
        relaxation_solves = relaxation_solves,
    )
end

function mean_summary(rows, rho, method)
    selected = filter(row -> row.rho == rho && row.method == method, rows)
    isempty(selected) && error("Missing Phase 14 rows for rho=$(rho), method=$(method).")
    finite_ess = filter(isfinite, [row.ess for row in selected])
    return (
        cosine = mean(row.cosine for row in selected),
        relative_error = mean(row.relative_error for row in selected),
        original_improvement = mean(row.original_improvement for row in selected),
        smoothed_improvement = mean(row.smoothed_improvement for row in selected),
        switch_rate = mean(row.branch_switched for row in selected),
        gradient_searches = sum(row.gradient_global_searches for row in selected),
        gradient_branch_solves = sum(row.gradient_branch_solves for row in selected),
        evaluation_searches = sum(row.evaluation_global_searches for row in selected),
        ess = isempty(finite_ess) ? NaN : mean(finite_ess),
    )
end

function paired_interval(values)
    length(values) >= 5 || return (mean = mean(values), lower = NaN, upper = NaN)
    center = mean(values)
    halfwidth = 2.776 * std(values; corrected = true) / sqrt(length(values))
    return (mean = center, lower = center - halfwidth, upper = center + halfwidth)
end

function main()
    profile = Symbol(lowercase(get(ENV, "SAGE_PHASE14_PROFILE", "smoke")))
    setup = configuration(profile)
    output_directory = abspath(get(
        ENV, "SAGE_PHASE14_OUTPUT", "phase14_mechanism_smoke_results",
    ))
    mkpath(output_directory)
    rows = NamedTuple[]
    reference_rows = NamedTuple[]
    maximum_violation = 0.0
    candidate_labels = ["reference", "hard_local", "ste"]
    append!(candidate_labels, ["sage_n$(samples)" for samples in setup.sample_counts])

    # Warm every path and dimensional specialization before diagnostic timings/work.
    warm_config = JobShopConfig(n_jobs = 3, n_machines = 4)
    warm = prepare_problem(
        999; config = warm_config, train_size = 2, iid_size = 2, ood_size = 2,
    )
    warm_network = pretrain_network(warm, 999; epochs = 1, batch_size = 2)
    for method in (:hard_local, :ste, :blackbox, :sage_multiscale)
        estimate_output_gradient(
            warm, warm_network, 1, method, 1_399_000 + Int(method == :ste);
            radius = 0.3, samples = 4,
        )
    end

    for seed in setup.seeds
        config = JobShopConfig(n_jobs = 3, n_machines = 4)
        problem = prepare_problem(
            seed;
            config = config, train_size = setup.train_size,
            iid_size = setup.iid_size, ood_size = setup.ood_size,
        )
        network = pretrain_network(
            problem, seed;
            epochs = setup.pretrain_epochs, batch_size = setup.pretrain_batch,
            learning_rate = 2e-3,
        )
        instance = problem.train.instances[1]

        for (rho_index, rho) in enumerate(setup.rhos)
            seed_base = 1_400_000_000 + 1_000_000seed + 10_000rho_index
            reference_a = estimate_output_gradient(
                problem, network, 1, :blackbox, seed_base + 1;
                radius = rho, samples = setup.reference_samples,
            )
            reference_b = estimate_output_gradient(
                problem, network, 1, :blackbox, seed_base + 2;
                radius = rho, samples = setup.reference_samples,
            )
            reference_gradient = (reference_a.gradient .+ reference_b.gradient) ./ 2
            reference_norm = norm(reference_gradient)
            reference_norm > 0 || error("Phase 14 averaged reference norm is zero.")
            replicate_cosine = dot(reference_a.gradient, reference_b.gradient) /
                max(norm(reference_a.gradient) * norm(reference_b.gradient), eps(Float64))
            push!(reference_rows, (
                seed = seed, rho = rho, replicate_cosine = replicate_cosine,
                reference_norm = reference_norm,
                replicate_a_norm = norm(reference_a.gradient),
                replicate_b_norm = norm(reference_b.gradient),
                samples_per_replicate = setup.reference_samples,
                global_searches = reference_a.global_searches +
                    reference_b.global_searches,
                branch_solves = reference_a.branch_solves + reference_b.branch_solves,
            ))

            hard = estimate_output_gradient(
                problem, network, 1, :hard_local, seed_base + 3;
                radius = rho, samples = 2,
            )
            ste = estimate_output_gradient(
                problem, network, 1, :ste, seed_base + 4;
                radius = rho, samples = 2,
            )
            candidates = [(
                label = "reference", gradient = reference_gradient,
                searches = reference_a.global_searches + reference_b.global_searches,
                branch_solves = reference_a.branch_solves + reference_b.branch_solves,
                ess = NaN, declared_samples = 2 * setup.reference_samples,
                proposal_samples = 0, unresolved = reference_a.unresolved +
                    reference_b.unresolved,
                logical_failures = reference_a.logical_failures +
                    reference_b.logical_failures,
            ), (
                label = "hard_local", gradient = hard.gradient,
                searches = hard.global_searches, branch_solves = hard.branch_solves,
                ess = hard.ess, declared_samples = 0, proposal_samples = 0,
                unresolved = hard.unresolved, logical_failures = hard.logical_failures,
            ), (
                label = "ste", gradient = ste.gradient,
                searches = ste.global_searches, branch_solves = ste.branch_solves,
                ess = ste.ess, declared_samples = 0, proposal_samples = 0,
                unresolved = ste.unresolved, logical_failures = ste.logical_failures,
            )]
            for samples in setup.sample_counts
                sage = estimate_output_gradient(
                    problem, network, 1, :sage_multiscale,
                    seed_base + 100 + samples;
                    radius = rho, samples = samples,
                )
                push!(candidates, (
                    label = "sage_n$(samples)", gradient = sage.gradient,
                    searches = sage.global_searches,
                    branch_solves = sage.branch_solves, ess = sage.ess,
                    declared_samples = samples,
                    proposal_samples = sage.base_samples + sage.local_samples +
                        sage.focused_samples,
                    unresolved = sage.unresolved,
                    logical_failures = sage.logical_failures,
                ))
            end

            evaluation_rng = MersenneTwister(seed_base + 9_000)
            evaluation_perturbations = sample_antithetic_ball_smoothing(
                evaluation_rng, length(reference_gradient), setup.evaluation_samples;
                beta = 2.0,
            )
            base_exact = exact_task_loss(instance, reference_a.point, config)
            base_smoothed = smoothed_task_loss(
                instance, reference_a.point, evaluation_perturbations, rho, config,
            )
            maximum_violation = max(
                maximum_violation, base_exact.violation, base_smoothed.violation,
            )

            for candidate in candidates
                gradient_norm = norm(candidate.gradient)
                gradient_norm > 0 || error(
                    "Phase 14 $(candidate.label) gradient norm is zero.",
                )
                direction = candidate.gradient ./ gradient_norm
                shifted = reference_a.point .- setup.step_size .* direction
                stepped = exact_task_loss(instance, shifted, config)
                stepped_smoothed = smoothed_task_loss(
                    instance, shifted, evaluation_perturbations, rho, config,
                )
                maximum_violation = max(
                    maximum_violation, stepped.violation, stepped_smoothed.violation,
                )
                cosine = dot(candidate.gradient, reference_gradient) /
                    max(gradient_norm * reference_norm, eps(Float64))
                relative_error = norm(candidate.gradient - reference_gradient) /
                    reference_norm
                push!(rows, (
                    seed = seed, rho = rho, method = candidate.label,
                    declared_samples = candidate.declared_samples,
                    cosine = cosine, relative_error = relative_error,
                    gradient_norm = gradient_norm, reference_norm = reference_norm,
                    reference_replicate_cosine = replicate_cosine,
                    step_size = setup.step_size,
                    base_original_loss = base_exact.loss,
                    stepped_original_loss = stepped.loss,
                    original_improvement = base_exact.loss - stepped.loss,
                    base_smoothed_loss = base_smoothed.loss,
                    stepped_smoothed_loss = stepped_smoothed.loss,
                    smoothed_improvement = base_smoothed.loss - stepped_smoothed.loss,
                    branch_switched = stepped.choices != base_exact.choices,
                    gradient_global_searches = candidate.searches,
                    gradient_branch_solves = candidate.branch_solves,
                    evaluation_global_searches = base_smoothed.searches +
                        stepped_smoothed.searches,
                    ess = candidate.ess,
                    proposal_samples = candidate.proposal_samples,
                    unresolved = candidate.unresolved,
                    logical_failures = candidate.logical_failures,
                    maximum_violation = max(
                        base_exact.violation, base_smoothed.violation,
                        stepped.violation, stepped_smoothed.violation,
                    ),
                ))
            end
            println(
                "[phase14] seed=$(seed) rho=$(rho) reference_cosine=" *
                "$(round(replicate_cosine; digits = 3))",
            )
            flush(stdout)
        end
    end

    rows_path = joinpath(output_directory, "phase14_one_step_rows.csv")
    reference_path = joinpath(output_directory, "phase14_reference_rows.csv")
    summary_path = joinpath(output_directory, "phase14_summary.json")
    write_csv(rows_path, propertynames(first(rows)), rows)
    write_csv(reference_path, propertynames(first(reference_rows)), reference_rows)

    expected_rows = length(setup.seeds) * length(setup.rhos) *
        length(candidate_labels)
    completeness_gate = length(rows) == expected_rows
    safety_gate = maximum_violation <= 1e-6 && all(
        row.unresolved == 0 && row.logical_failures == 0 &&
        row.maximum_violation <= 1e-6 for row in rows
    )
    reference_gate = all(
        isfinite(row.replicate_cosine) && isfinite(row.reference_norm) &&
        row.reference_norm > 0 && isfinite(row.replicate_a_norm) &&
        isfinite(row.replicate_b_norm) for row in reference_rows
    )
    finite_gate = all(
        isfinite(row.cosine) && isfinite(row.relative_error) &&
        isfinite(row.gradient_norm) && row.gradient_norm > 0 &&
        isfinite(row.original_improvement) && isfinite(row.smoothed_improvement)
        for row in rows
    )
    sage_rows = filter(row -> startswith(row.method, "sage_n"), rows)
    sage_gate = all(
        isfinite(row.ess) && row.ess > 0 &&
        row.proposal_samples in (0, row.declared_samples)
        for row in sage_rows
    )

    summaries = Dict{String, Any}()
    tolerance = 1e-5
    objective_votes = 0
    variance_votes = 0
    for rho in setup.rhos
        rho_key = string(rho)
        summaries[rho_key] = Dict{String, Any}()
        for method in candidate_labels
            item = mean_summary(rows, rho, method)
            values = filter(row -> row.rho == rho && row.method == method, rows)
            original_interval = paired_interval([
                row.original_improvement for row in values
            ])
            smoothed_interval = paired_interval([
                row.smoothed_improvement for row in values
            ])
            summaries[rho_key][method] = Dict(
                "mean_cosine" => item.cosine,
                "mean_relative_error" => item.relative_error,
                "mean_original_improvement" => item.original_improvement,
                "original_improvement_ci95" => [
                    original_interval.lower, original_interval.upper,
                ],
                "mean_smoothed_improvement" => item.smoothed_improvement,
                "smoothed_improvement_ci95" => [
                    smoothed_interval.lower, smoothed_interval.upper,
                ],
                "branch_switch_rate" => item.switch_rate,
                "gradient_global_searches" => item.gradient_searches,
                "gradient_branch_solves" => item.gradient_branch_solves,
                "evaluation_global_searches" => item.evaluation_searches,
                "mean_ess" => item.ess,
            )
        end
        reference = mean_summary(rows, rho, "reference")
        sage_4 = mean_summary(rows, rho, "sage_n4")
        largest_n = maximum(setup.sample_counts)
        sage_large = mean_summary(rows, rho, "sage_n$(largest_n)")
        objective_votes += reference.smoothed_improvement > tolerance &&
            reference.original_improvement <= tolerance
        variance_votes += reference.original_improvement > tolerance &&
            sage_large.original_improvement > sage_4.original_improvement + tolerance
    end
    required_votes = min(2, length(setup.rhos))
    objective_mismatch = objective_votes >= required_votes
    estimator_variance = variance_votes >= required_votes
    diagnosis = objective_mismatch && estimator_variance ? "BOTH" :
        objective_mismatch ? "OBJECTIVE_MISMATCH" :
        estimator_variance ? "ESTIMATOR_VARIANCE" : "NEITHER_CONFIRMED"
    phase14_pass = profile == :pilot && completeness_gate && safety_gate &&
        reference_gate && finite_gate && sage_gate

    summary = Dict{String, Any}(
        "phase" => "phase14_mechanism_diagnosis",
        "profile" => string(profile),
        "status" => profile == :smoke ? "SMOKE_COMPLETE" :
            phase14_pass ? "PASS" : "FAIL",
        "diagnosis" => diagnosis,
        "objective_mismatch_votes" => objective_votes,
        "estimator_variance_votes" => variance_votes,
        "seeds" => setup.seeds, "rhos" => setup.rhos,
        "sample_counts" => setup.sample_counts,
        "reference_samples_per_replicate" => setup.reference_samples,
        "evaluation_samples" => setup.evaluation_samples,
        "step_size" => setup.step_size,
        "results" => summaries,
        "maximum_violation" => maximum_violation,
        "completeness_gate" => completeness_gate,
        "safety_gate" => safety_gate, "reference_gate" => reference_gate,
        "finite_gate" => finite_gate, "sage_gate" => sage_gate,
        "pass" => phase14_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 14 objective-versus-variance diagnosis ", profile)
    println("Julia version: ", VERSION)
    println("Retained contract: exact forward, compact q, fixed rho per slice, Eq.(11)-(12),(17)-(21), exact q/p")
    println("Diagnostic additions: rho/N sweep, averaged reference, normalized step, common-random objective evaluation")
    println("method\trho\tcosine\trelative error\toriginal improvement\tsmoothed improvement\tswitch rate\tESS")
    for rho in setup.rhos, method in candidate_labels
        item = mean_summary(rows, rho, method)
        println("$(method)\t$(rho)\t$(item.cosine)\t$(item.relative_error)\t" *
            "$(item.original_improvement)\t$(item.smoothed_improvement)\t" *
            "$(item.switch_rate)\t$(item.ess)")
    end
    println("reference replicate cosine by rho: ")
    for rho in setup.rhos
        selected = filter(row -> row.rho == rho, reference_rows)
        println("  rho=$(rho): ", mean(row.replicate_cosine for row in selected))
    end
    println("objective mismatch/variance votes: ", objective_votes, "/", variance_votes)
    println("diagnosis: ", diagnosis)
    println("maximum logical violation: ", maximum_violation)
    for (name, passed) in (
        ("completeness", completeness_gate), ("safety", safety_gate),
        ("reference", reference_gate), ("finite metrics", finite_gate),
        ("SAGE accounting", sage_gate),
    )
        println(name, " gate: ", passed ? "PASS" : "FAIL")
    end
    println("row CSV: ", rows_path)
    println("reference CSV: ", reference_path)
    println("summary JSON: ", summary_path)
    println(profile == :smoke ? "PHASE14_MECHANISM_SMOKE=COMPLETE" :
        "PHASE14_MECHANISM_DIAGNOSIS=" * (phase14_pass ? "PASS" : "FAIL"))
    profile == :pilot && println("PHASE14_DIAGNOSIS=", diagnosis)
    profile == :pilot && !phase14_pass && exit(1)
end

main()
