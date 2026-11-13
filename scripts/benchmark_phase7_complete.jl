using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "phase7_scaling_common.jl"))
using .Phase7Scaling

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

cosine_similarity(left, right) = dot(left, right) / (norm(left) * norm(right))

function configuration_specs()
    # Controlled one-factor slices plus the largest joint stress case.
    return [
        (dimension = 2, branches = 8, slice = "dimension"),
        (dimension = 4, branches = 8, slice = "dimension+branch"),
        (dimension = 8, branches = 8, slice = "dimension"),
        (dimension = 4, branches = 4, slice = "branch"),
        (dimension = 4, branches = 16, slice = "branch"),
        (dimension = 8, branches = 16, slice = "joint_stress"),
    ]
end

function switching_point(geometry)
    left = geometry.centers[:, 1]
    right = geometry.centers[:, 2]
    return (
        z = 0.52 .* left .+ 0.48 .* right,
        radius = 0.40 * norm(left .- right),
    )
end

function workload_centers(geometry)
    centers = geometry.centers
    branches = branch_count(geometry)
    separation = minimum_center_distance(geometry)
    return [
        (slice = "stable", center = copy(centers[:, 1]), radius = 0.10 * separation),
        (slice = "stable", center = copy(centers[:, branches]), radius = 0.10 * separation),
        (
            slice = "switching",
            center = 0.52 .* centers[:, 1] .+ 0.48 .* centers[:, 2],
            radius = 0.40 * norm(centers[:, 1] .- centers[:, 2]),
        ),
        (
            slice = "switching",
            center = 0.52 .* centers[:, branches - 1] .+ 0.48 .* centers[:, branches],
            radius = 0.40 * norm(
                centers[:, branches - 1] .- centers[:, branches],
            ),
        ),
    ]
end

function certified_sage_diagnostics(
    model, geometry, center_result, z, radius, target, samples,
)
    started = time_ns()
    estimate = sage_gradient(
        model,
        z,
        radius,
        y -> task_loss(y, target),
        y -> task_loss_gradient(y, target);
        samples = samples,
        center_result = center_result,
        store_records = true,
    )
    seconds = (time_ns() - started) / 1e9
    estimate.status == :ok || error("Certified SAGE returned $(estimate.status).")
    unsafe = 0
    feasible = true
    for record in estimate.records
        exact = analytic_project(geometry, record.w)
        unsafe += only(record.global_choices) != exact.branch ||
            !isapprox(record.global_y, exact.y; atol = 5e-7)
        feasible &= record.status == :ok &&
            check_feasibility(model, record.global_y).feasible
    end
    return (estimate = estimate, unsafe = unsafe, feasible = feasible, seconds = seconds)
end

function main()
    repetitions = parse(Int, get(ENV, "SAGE_PHASE7_COMPLETE_REPETITIONS", "20"))
    samples_per_dimension = parse(
        Int, get(ENV, "SAGE_PHASE7_COMPLETE_SAMPLES_PER_DIMENSION", "128"),
    )
    reference_samples = parse(
        Int, get(ENV, "SAGE_PHASE7_COMPLETE_REFERENCE_SAMPLES", "100000"),
    )
    backend_samples = parse(Int, get(ENV, "SAGE_PHASE7_COMPLETE_BACKEND_SAMPLES", "8"))
    timing_repetitions = parse(
        Int, get(ENV, "SAGE_PHASE7_COMPLETE_TIMING_REPETITIONS", "5"),
    )
    output_directory = abspath(get(
        ENV, "SAGE_PHASE7_COMPLETE_OUTPUT", "phase7_complete_results",
    ))
    repetitions >= 5 || error("At least five statistical repetitions are required.")
    samples_per_dimension > 0 || error("Samples per dimension must be positive.")
    reference_samples > 0 || error("Reference sample count must be positive.")
    iseven(reference_samples) || error("Reference sample count must be even.")
    reference_samples >= 8 * samples_per_dimension ||
        error("Reference budget must cover the largest dimension-aware estimate budget.")
    backend_samples > 0 || error("Backend sample count must be positive.")
    iseven(backend_samples) || error("Backend sample count must be even.")
    timing_repetitions >= 3 || error("At least three timing repetitions are required.")
    mkpath(output_directory)

    reference_rows = NamedTuple[]
    fidelity_rows = NamedTuple[]
    references = Dict{Tuple{Int, Int}, Vector{Float64}}()

    # Two independent high-budget estimates make reference convergence visible;
    # their average is the evaluation reference, never an analytic closed form.
    for spec in configuration_specs()
        d, branches = spec.dimension, spec.branches
        geometry = scaling_box_union(d, branches)
        target = copy(geometry.centers[:, branches])
        point = switching_point(geometry)
        reference_estimates = Vector{Vector{Float64}}()
        for replicate in 1:2
            samples = sample_matrix(
                MersenneTwister(20700000 + 10000d + 100branches + replicate),
                d,
                reference_samples;
                antithetic = true,
            )
            push!(reference_estimates, analytic_sage_gradient(
                geometry, point.z, point.radius, target, samples,
            ).gradient)
        end
        reference = (reference_estimates[1] .+ reference_estimates[2]) ./ 2
        norm(reference) > 1e-8 || error("Reference gradient is numerically zero.")
        disagreement = norm(reference_estimates[1] .- reference_estimates[2]) /
            norm(reference)
        cosine = cosine_similarity(reference_estimates[1], reference_estimates[2])
        references[(d, branches)] = reference
        push!(reference_rows, (
            dimension = d,
            branches = branches,
            slice = spec.slice,
            samples_per_replicate = reference_samples,
            relative_disagreement = disagreement,
            cosine = cosine,
        ))

        estimate_samples = samples_per_dimension * d
        iseven(estimate_samples) || error("Dimension-aware sample count must be even.")
        estimates = Dict(
            "hard_local" => Vector{Vector{Float64}}(),
            "ste" => Vector{Vector{Float64}}(),
            "blackbox" => Vector{Vector{Float64}}(),
            "sage" => Vector{Vector{Float64}}(),
        )
        for repetition in 1:repetitions
            samples = sample_matrix(
                MersenneTwister(20710000 + 100000d + 1000branches + repetition),
                d,
                estimate_samples;
                antithetic = true,
            )
            push!(estimates["hard_local"], hard_local_gradient(geometry, point.z, target))
            push!(estimates["ste"], ste_gradient(geometry, point.z, target))
            push!(estimates["blackbox"], analytic_blackbox_gradient(
                geometry, point.z, point.radius, target, samples,
            ))
            push!(estimates["sage"], analytic_sage_gradient(
                geometry, point.z, point.radius, target, samples,
            ).gradient)
        end
        for method in ("hard_local", "ste", "blackbox", "sage")
            summary = estimator_summary(estimates[method], reference)
            push!(fidelity_rows, (
                dimension = d,
                branches = branches,
                slice = spec.slice,
                method = method,
                repetitions = repetitions,
                samples = estimate_samples,
                relative_bias = summary.relative_bias,
                trace_variance = summary.trace_variance,
                rmse = summary.rmse,
                mean_cosine = summary.mean_cosine,
                minimum_cosine = summary.minimum_cosine,
            ))
        end
    end

    workload_rows = NamedTuple[]
    maximum_projection_parity_error = 0.0
    maximum_gradient_parity_error = 0.0
    unsafe_pruning = 0
    all_feasible = true
    blackbox_times = Float64[]
    sage_times = Float64[]

    # Backend profiling uses representative small, branch-scaling, and joint
    # stress cases. Each repetition reuses identical perturbations per method.
    profile_specs = [(2, 8), (4, 16), (8, 16)]
    for (d, branches) in profile_specs
        geometry = scaling_box_union(d, branches)
        model = disjunctive_model(geometry)
        target = copy(geometry.centers[:, branches])
        centers = workload_centers(geometry)

        # Compile all relevant backend paths outside recorded timing.
        warm = centers[1]
        warm_center = lazy_branch_search(model, warm.center; store_prune_records = false)
        warm_samples = sample_matrix(MersenneTwister(1), d, 2; antithetic = true)
        backend_blackbox_gradient(
            model, geometry, warm_center, warm.center, warm.radius, target, warm_samples,
        )
        certified_sage_diagnostics(
            model, geometry, warm_center, warm.center, warm.radius, target, warm_samples,
        )

        for repetition in 1:timing_repetitions
            repetition_blackbox_seconds = 0.0
            repetition_sage_seconds = 0.0
            for (center_index, item) in enumerate(centers)
                samples = sample_matrix(
                    MersenneTwister(
                        20720000 + 100000d + 1000branches + 10repetition + center_index,
                    ),
                    d,
                    backend_samples;
                    antithetic = true,
                )
                center_result = lazy_branch_search(
                    model, item.center; store_prune_records = false,
                )
                center_result.status == :optimal || error("Workload center was unresolved.")
                exact_center = analytic_project(geometry, item.center)
                projection_error = norm(center_result.y .- exact_center.y)
                maximum_projection_parity_error = max(
                    maximum_projection_parity_error, projection_error,
                )

                # Alternate order to reduce systematic first-method timing bias.
                if isodd(repetition)
                    blackbox = backend_blackbox_gradient(
                        model, geometry, center_result, item.center, item.radius,
                        target, samples,
                    )
                    sage = certified_sage_diagnostics(
                        model, geometry, center_result, item.center, item.radius,
                        target, samples,
                    )
                else
                    sage = certified_sage_diagnostics(
                        model, geometry, center_result, item.center, item.radius,
                        target, samples,
                    )
                    blackbox = backend_blackbox_gradient(
                        model, geometry, center_result, item.center, item.radius,
                        target, samples,
                    )
                end
                repetition_blackbox_seconds += blackbox.seconds
                repetition_sage_seconds += sage.seconds

                if repetition == 1
                    expected_blackbox = analytic_blackbox_gradient(
                        geometry, item.center, item.radius, target, samples,
                    )
                    expected_sage = analytic_sage_gradient(
                        geometry, item.center, item.radius, target, samples,
                    ).gradient
                    blackbox_error = norm(blackbox.gradient .- expected_blackbox)
                    sage_error = norm(sage.estimate.gradient .- expected_sage)
                    maximum_gradient_parity_error = max(
                        maximum_gradient_parity_error, blackbox_error, sage_error,
                    )
                    unsafe_pruning += blackbox.unsafe_pruning + sage.unsafe
                    all_feasible &= blackbox.feasible && sage.feasible
                    certificate = sage.estimate.certificate
                    pruning_rate = certificate.total_competitor_branches == 0 ? 1.0 :
                        certificate.stable_pruned_branches /
                        certificate.total_competitor_branches
                    for (method, global_searches, reference_solves, competitor_solves,
                         restricted_searches, lazy_searches, leaf_solves, relaxation_solves,
                         parity_error, seconds) in (
                        (
                            "blackbox", blackbox.global_searches,
                            blackbox.reference_branch_solves,
                            blackbox.competitor_branch_solves, 0,
                            blackbox.global_searches, blackbox.leaf_solves,
                            blackbox.relaxation_solves, blackbox_error, blackbox.seconds,
                        ),
                        (
                            "sage", sage.estimate.perturbed_global_searches,
                            sage.estimate.reference_branch_solves,
                            sage.estimate.competitor_branch_solves,
                            sage.estimate.restricted_global_searches,
                            sage.estimate.lazy_global_searches,
                            sage.estimate.lazy_leaf_solves,
                            sage.estimate.lazy_relaxation_solves,
                            sage_error, sage.seconds,
                        ),
                    )
                        push!(workload_rows, (
                            dimension = d,
                            branches = branches,
                            slice = item.slice,
                            center = center_index,
                            method = method,
                            samples = backend_samples,
                            certificate_status = certificate.status,
                            competitor_pruning_rate = pruning_rate,
                            global_searches = global_searches,
                            reference_branch_solves = reference_solves,
                            competitor_branch_solves = competitor_solves,
                            restricted_global_searches = restricted_searches,
                            lazy_global_searches = lazy_searches,
                            leaf_solves = leaf_solves,
                            relaxation_solves = relaxation_solves,
                            gradient_parity_error = parity_error,
                            seconds = seconds,
                        ))
                    end
                end
            end
            push!(blackbox_times, repetition_blackbox_seconds)
            push!(sage_times, repetition_sage_seconds)
        end
    end

    reference_gate = all(
        row.cosine >= 0.99 && row.relative_disagreement <= 0.15
        for row in reference_rows
    )
    sage_rows = filter(row -> row.method == "sage", fidelity_rows)
    blackbox_rows = filter(row -> row.method == "blackbox", fidelity_rows)
    fidelity_gate = all(
        row.mean_cosine >= 0.90 && row.relative_bias <= 0.20
        for row in sage_rows
    )
    rmse_gate = all(begin
        blackbox = only(filter(
            candidate -> candidate.dimension == row.dimension &&
                candidate.branches == row.branches,
            blackbox_rows,
        ))
        row.rmse <= 1.10 * blackbox.rmse
    end for row in sage_rows)
    aggregate_variance_ratio = sum(row.trace_variance for row in sage_rows) /
        sum(row.trace_variance for row in blackbox_rows)
    aggregate_rmse_ratio = sqrt(
        sum(row.rmse^2 for row in sage_rows) /
        sum(row.rmse^2 for row in blackbox_rows),
    )

    blackbox_global_searches = sum(
        row.global_searches for row in workload_rows if row.method == "blackbox"
    )
    sage_global_searches = sum(
        row.global_searches for row in workload_rows if row.method == "sage"
    )
    global_search_reduction = 1.0 - sage_global_searches / blackbox_global_searches
    median_blackbox_time = median(blackbox_times)
    median_sage_time = median(sage_times)
    wall_time_ratio = median_sage_time / median_blackbox_time
    parity_gate = maximum_projection_parity_error < 1e-6 &&
        maximum_gradient_parity_error < 1e-5
    safe_pruning_gate = unsafe_pruning == 0 && all_feasible
    global_reduction_gate = global_search_reduction >= 0.25
    variance_efficiency_gate = aggregate_variance_ratio <= 0.95
    wall_time_gate = wall_time_ratio <= 1.0
    # Proposal roadmap criterion: advance when variance OR total computation
    # improves under a matched perturbation budget. Both are still reported.
    efficiency_gate = variance_efficiency_gate || wall_time_gate
    phase7_pass = reference_gate && fidelity_gate && rmse_gate && parity_gate &&
        safe_pruning_gate && global_reduction_gate && efficiency_gate

    reference_path = joinpath(output_directory, "phase7_reference_convergence.csv")
    fidelity_path = joinpath(output_directory, "phase7_fidelity.csv")
    workload_path = joinpath(output_directory, "phase7_workload.csv")
    summary_path = joinpath(output_directory, "phase7_summary.json")
    write_csv(reference_path, (
        :dimension, :branches, :slice, :samples_per_replicate,
        :relative_disagreement, :cosine,
    ), reference_rows)
    write_csv(fidelity_path, (
        :dimension, :branches, :slice, :method, :repetitions, :samples,
        :relative_bias, :trace_variance, :rmse, :mean_cosine, :minimum_cosine,
    ), fidelity_rows)
    write_csv(workload_path, (
        :dimension, :branches, :slice, :center, :method, :samples,
        :certificate_status, :competitor_pruning_rate, :global_searches,
        :reference_branch_solves, :competitor_branch_solves,
        :restricted_global_searches, :lazy_global_searches, :leaf_solves,
        :relaxation_solves, :gradient_parity_error, :seconds,
    ), workload_rows)
    summary = Dict{String, Any}(
        "phase" => "phase7_complete",
        "julia_version" => string(VERSION),
        "sampling_scheme" => "antithetic_compact_q_beta_2",
        "repetitions" => repetitions,
        "samples_per_dimension" => samples_per_dimension,
        "reference_samples_per_replicate" => reference_samples,
        "backend_samples" => backend_samples,
        "timing_repetitions" => timing_repetitions,
        "maximum_projection_parity_error" => maximum_projection_parity_error,
        "maximum_gradient_parity_error" => maximum_gradient_parity_error,
        "unsafe_pruning" => unsafe_pruning,
        "aggregate_variance_ratio" => aggregate_variance_ratio,
        "aggregate_rmse_ratio" => aggregate_rmse_ratio,
        "global_search_reduction" => global_search_reduction,
        "median_blackbox_seconds" => median_blackbox_time,
        "median_sage_seconds" => median_sage_time,
        "wall_time_ratio" => wall_time_ratio,
        "reference_gate" => reference_gate,
        "fidelity_gate" => fidelity_gate,
        "rmse_gate" => rmse_gate,
        "parity_gate" => parity_gate,
        "safe_pruning_gate" => safe_pruning_gate,
        "global_reduction_gate" => global_reduction_gate,
        "variance_efficiency_gate" => variance_efficiency_gate,
        "wall_time_gate" => wall_time_gate,
        "efficiency_gate" => efficiency_gate,
        "pass" => phase7_pass,
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 7 complete controlled scaling gate")
    println("Julia version: ", VERSION)
    println("Proposal contract: exact feasible forward, compact q beta=2, fixed rho per slice, Eq.(17)-(18)")
    println("Sampling: antithetic pairs with unchanged compact-q marginals")
    println("repetitions/samples-per-dimension/reference/backend/timing: ",
        repetitions, "/", samples_per_dimension, "/", reference_samples, "/",
        backend_samples, "/", timing_repetitions)
    println()
    println("configuration\tN\tref disagreement\tSAGE bias\tSAGE RMSE\tblack-box RMSE\tSAGE cosine")
    for row in sage_rows
        reference_row = only(filter(
            candidate -> candidate.dimension == row.dimension &&
                candidate.branches == row.branches,
            reference_rows,
        ))
        blackbox = only(filter(
            candidate -> candidate.dimension == row.dimension &&
                candidate.branches == row.branches,
            blackbox_rows,
        ))
        println("d=$(row.dimension),K=$(row.branches)\t$(row.samples)\t",
            "$(reference_row.relative_disagreement)\t$(row.relative_bias)\t",
            "$(row.rmse)\t$(blackbox.rmse)\t$(row.mean_cosine)")
    end
    println()
    println("aggregate SAGE/black-box variance ratio: ", aggregate_variance_ratio)
    println("aggregate SAGE/black-box RMSE ratio: ", aggregate_rmse_ratio)
    println("unsafe pruning count: ", unsafe_pruning)
    println("maximum projection/backend-gradient parity error: ",
        maximum_projection_parity_error, "/", maximum_gradient_parity_error)
    println("black-box/SAGE global searches: ",
        blackbox_global_searches, "/", sage_global_searches)
    println("global-search reduction: ", global_search_reduction)
    println("median black-box/SAGE seconds: ",
        median_blackbox_time, "/", median_sage_time)
    println("median wall-time ratio: ", wall_time_ratio)
    println("reference convergence gate: ", reference_gate ? "PASS" : "FAIL")
    println("fidelity gate: ", fidelity_gate ? "PASS" : "FAIL")
    println("per-configuration RMSE gate: ", rmse_gate ? "PASS" : "FAIL")
    println("parity gate: ", parity_gate ? "PASS" : "FAIL")
    println("safe-pruning/feasibility gate: ", safe_pruning_gate ? "PASS" : "FAIL")
    println("global-search reduction gate: ", global_reduction_gate ? "PASS" : "FAIL")
    println("variance efficiency gate: ", variance_efficiency_gate ? "PASS" : "FAIL")
    println("wall-time gate: ", wall_time_gate ? "PASS" : "FAIL")
    println("proposal variance-or-time efficiency gate: ", efficiency_gate ? "PASS" : "FAIL")
    println("reference CSV: ", reference_path)
    println("fidelity CSV: ", fidelity_path)
    println("workload CSV: ", workload_path)
    println("summary JSON: ", summary_path)
    println("PHASE7_COMPLETE=", phase7_pass ? "PASS" : "FAIL")
end

main()
