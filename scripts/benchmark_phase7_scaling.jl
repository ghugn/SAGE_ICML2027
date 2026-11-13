using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "phase7_scaling_common.jl"))
using .Phase7Scaling

function parse_integer_list(name, default)
    raw = get(ENV, name, default)
    values = parse.(Int, strip.(split(raw, ',')))
    isempty(values) && error("$(name) cannot be empty.")
    all(>(0), values) || error("$(name) must contain positive integers.")
    return values
end

function csv_value(value)
    if value isa AbstractString || value isa Symbol
        text = replace(string(value), '"' => "\"\"")
        return occursin(r"[,\"\n]", text) ? "\"$(text)\"" : text
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
        string(value),
        '\\' => "\\\\",
        '"' => "\\\"",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
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

function full_sage_diagnostics(model, geometry, center_result, z, radius, target, samples)
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

function hard_local_backend(model, center_result, z, target)
    started = time_ns()
    reference = project_branch(model, z, center_result.choices)
    local_result = fixed_branch_loss_gradient(
        model,
        z,
        center_result.choices,
        y -> task_loss_gradient(y, target);
        branch_result = reference,
    )
    seconds = (time_ns() - started) / 1e9
    local_result.status == :ok || error("Hard-local derivative was unresolved.")
    return (gradient = local_result.gradient, seconds = seconds)
end

function ste_backend(center_result, target)
    started = time_ns()
    gradient = task_loss_gradient(center_result.y, target)
    seconds = (time_ns() - started) / 1e9
    return (gradient = gradient, seconds = seconds)
end

function workload_centers(geometry)
    centers = geometry.centers
    branches = branch_count(geometry)
    minimum_distance = minimum_center_distance(geometry)
    stable_radius = 0.10 * minimum_distance
    stable = [
        (slice = "stable", center = copy(centers[:, 1]), radius = stable_radius),
        (slice = "stable", center = copy(centers[:, branches]), radius = stable_radius),
    ]
    pairs = ((1, 2), (branches - 1, branches))
    switching = [
        (
            slice = "switching",
            center = 0.52 .* centers[:, left] .+ 0.48 .* centers[:, right],
            radius = 0.40 * norm(centers[:, left] .- centers[:, right]),
        )
        for (left, right) in pairs
    ]
    return vcat(stable, switching)
end

function main()
    dimensions = parse_integer_list("SAGE_PHASE7_DIMENSIONS", "2,4,8")
    branch_counts = parse_integer_list("SAGE_PHASE7_BRANCHES", "4,8,16")
    repetitions = parse(Int, get(ENV, "SAGE_PHASE7_REPETITIONS", "5"))
    estimate_samples = parse(Int, get(ENV, "SAGE_PHASE7_SAMPLES", "64"))
    reference_samples = parse(Int, get(ENV, "SAGE_PHASE7_REFERENCE_SAMPLES", "20000"))
    backend_samples = parse(Int, get(ENV, "SAGE_PHASE7_BACKEND_SAMPLES", "8"))
    sampling_scheme = Symbol(get(ENV, "SAGE_PHASE7_SAMPLING", "iid"))
    sampling_scheme in (:iid, :antithetic) ||
        error("SAGE_PHASE7_SAMPLING must be iid or antithetic.")
    antithetic_sampling = sampling_scheme == :antithetic
    output_directory = abspath(get(ENV, "SAGE_PHASE7_OUTPUT", "phase7_results"))
    repetitions > 1 || error("SAGE_PHASE7_REPETITIONS must exceed one.")
    estimate_samples > 0 || error("SAGE_PHASE7_SAMPLES must be positive.")
    reference_samples >= estimate_samples ||
        error("Reference sample count must be at least the estimate sample count.")
    backend_samples > 0 || error("SAGE_PHASE7_BACKEND_SAMPLES must be positive.")
    mkpath(output_directory)

    fidelity_rows = NamedTuple[]
    workload_rows = NamedTuple[]
    maximum_projection_parity_error = 0.0
    maximum_backend_gradient_parity_error = 0.0

    for d in dimensions, branches in branch_counts
        geometry = scaling_box_union(d, branches)
        model = disjunctive_model(geometry)
        target = copy(geometry.centers[:, branches])
        left = geometry.centers[:, 1]
        right = geometry.centers[:, 2]
        fidelity_z = 0.52 .* left .+ 0.48 .* right
        fidelity_radius = 0.40 * norm(left .- right)

        # Analytic/backend projection parity ties the cheap scaling family to
        # the certified DisjunctiveNet exact-forward implementation.
        for point in (copy(left), copy(target), fidelity_z)
            analytic = analytic_project(geometry, point)
            backend = lazy_branch_search(model, point; store_prune_records = false)
            backend.status == :optimal || error("Projection parity backend was unresolved.")
            only(backend.choices) == analytic.branch || error("Projection branch parity failed.")
            parity_error = norm(backend.y .- analytic.y)
            maximum_projection_parity_error = max(maximum_projection_parity_error, parity_error)
            parity_error < 1e-6 || error("Projection parity tolerance exceeded.")
            backend.report !== nothing && backend.report.feasible ||
                error("Projection parity output was logically infeasible.")
        end

        reference_matrix = sample_matrix(
            MersenneTwister(20270000 + 100d + branches),
            d,
            reference_samples,
            antithetic = antithetic_sampling,
        )
        reference = analytic_sage_gradient(
            geometry,
            fidelity_z,
            fidelity_radius,
            target,
            reference_matrix,
        ).gradient
        norm(reference) > 1e-8 || error("Reference gradient is numerically zero.")

        estimates = Dict(
            "hard_local" => Vector{Vector{Float64}}(),
            "ste" => Vector{Vector{Float64}}(),
            "blackbox" => Vector{Vector{Float64}}(),
            "sage_no_certificate" => Vector{Vector{Float64}}(),
            "sage_full" => Vector{Vector{Float64}}(),
        )
        for repetition in 1:repetitions
            samples = sample_matrix(
                MersenneTwister(20271000 + 10_000d + 100branches + repetition),
                d,
                estimate_samples,
                antithetic = antithetic_sampling,
            )
            hard = hard_local_gradient(geometry, fidelity_z, target)
            ste = ste_gradient(geometry, fidelity_z, target)
            blackbox = analytic_blackbox_gradient(
                geometry,
                fidelity_z,
                fidelity_radius,
                target,
                samples,
            )
            sage = analytic_sage_gradient(
                geometry,
                fidelity_z,
                fidelity_radius,
                target,
                samples,
            ).gradient
            push!(estimates["hard_local"], hard)
            push!(estimates["ste"], ste)
            push!(estimates["blackbox"], blackbox)
            # The certificate changes computation, never the estimator target.
            push!(estimates["sage_no_certificate"], copy(sage))
            push!(estimates["sage_full"], copy(sage))
        end

        for method in ("hard_local", "ste", "blackbox", "sage_no_certificate", "sage_full")
            summary = estimator_summary(estimates[method], reference)
            push!(fidelity_rows, (
                dimension = d,
                branches = branches,
                method = method,
                radius = fidelity_radius,
                repetitions = repetitions,
                samples = estimate_samples,
                reference_samples = reference_samples,
                relative_bias = summary.relative_bias,
                trace_variance = summary.trace_variance,
                rmse = summary.rmse,
                mean_cosine = summary.mean_cosine,
                minimum_cosine = summary.minimum_cosine,
            ))
        end

        centers = workload_centers(geometry)

        # Warm every backend path once so compilation does not decide the gate.
        warm = centers[1]
        warm_samples = sample_matrix(
            MersenneTwister(7), d, 2; antithetic = antithetic_sampling,
        )
        warm_center = lazy_branch_search(model, warm.center; store_prune_records = false)
        hard_local_backend(model, warm_center, warm.center, target)
        ste_backend(warm_center, target)
        backend_blackbox_gradient(
            model, geometry, warm_center, warm.center, warm.radius, target, warm_samples,
        )
        backend_sage_unpruned_gradient(
            model, geometry, warm_center, warm.center, warm.radius, target, warm_samples,
        )
        full_sage_diagnostics(
            model, geometry, warm_center, warm.center, warm.radius, target, warm_samples,
        )

        for (center_index, item) in enumerate(centers)
            samples = sample_matrix(
                MersenneTwister(20272000 + 10_000d + 100branches + center_index),
                d,
                backend_samples,
                antithetic = antithetic_sampling,
            )
            center_result = lazy_branch_search(
                model,
                item.center;
                store_prune_records = false,
            )
            center_result.status == :optimal || error("Workload center was unresolved.")
            certificate = switching_certificate(
                model,
                item.center,
                item.radius;
                center_result = center_result,
            )
            pruning_rate = certificate.total_competitor_branches == 0 ? 1.0 :
                certificate.stable_pruned_branches / certificate.total_competitor_branches

            hard = hard_local_backend(model, center_result, item.center, target)
            hard_expected = hard_local_gradient(geometry, item.center, target)
            hard_parity = norm(hard.gradient .- hard_expected)
            maximum_backend_gradient_parity_error = max(
                maximum_backend_gradient_parity_error,
                hard_parity,
            )
            push!(workload_rows, (
                dimension = d, branches = branches, slice = item.slice,
                center = center_index, method = "hard_local", radius = item.radius,
                samples = 1, certificate_status = certificate.status,
                competitor_pruning_rate = pruning_rate, global_searches = 0,
                reference_branch_solves = 1, competitor_branch_solves = 0,
                unsafe_pruning = 0, logically_feasible = true,
                gradient_parity_error = hard_parity, seconds = hard.seconds,
            ))

            ste = ste_backend(center_result, target)
            ste_expected = ste_gradient(geometry, item.center, target)
            ste_parity = norm(ste.gradient .- ste_expected)
            maximum_backend_gradient_parity_error = max(
                maximum_backend_gradient_parity_error,
                ste_parity,
            )
            push!(workload_rows, (
                dimension = d, branches = branches, slice = item.slice,
                center = center_index, method = "ste", radius = item.radius,
                samples = 1, certificate_status = certificate.status,
                competitor_pruning_rate = pruning_rate, global_searches = 0,
                reference_branch_solves = 0, competitor_branch_solves = 0,
                unsafe_pruning = 0, logically_feasible = true,
                gradient_parity_error = ste_parity, seconds = ste.seconds,
            ))

            blackbox = backend_blackbox_gradient(
                model,
                geometry,
                center_result,
                item.center,
                item.radius,
                target,
                samples,
            )
            blackbox_expected = analytic_blackbox_gradient(
                geometry,
                item.center,
                item.radius,
                target,
                samples,
            )
            blackbox_parity = norm(blackbox.gradient .- blackbox_expected)
            maximum_backend_gradient_parity_error = max(
                maximum_backend_gradient_parity_error,
                blackbox_parity,
            )
            push!(workload_rows, (
                dimension = d, branches = branches, slice = item.slice,
                center = center_index, method = "blackbox", radius = item.radius,
                samples = backend_samples, certificate_status = certificate.status,
                competitor_pruning_rate = pruning_rate,
                global_searches = blackbox.global_searches,
                reference_branch_solves = blackbox.reference_branch_solves,
                competitor_branch_solves = blackbox.competitor_branch_solves,
                unsafe_pruning = blackbox.unsafe_pruning,
                logically_feasible = blackbox.feasible,
                gradient_parity_error = blackbox_parity, seconds = blackbox.seconds,
            ))

            unpruned = backend_sage_unpruned_gradient(
                model,
                geometry,
                center_result,
                item.center,
                item.radius,
                target,
                samples,
            )
            sage_expected = analytic_sage_gradient(
                geometry,
                item.center,
                item.radius,
                target,
                samples,
            ).gradient
            unpruned_parity = norm(unpruned.gradient .- sage_expected)
            maximum_backend_gradient_parity_error = max(
                maximum_backend_gradient_parity_error,
                unpruned_parity,
            )
            push!(workload_rows, (
                dimension = d, branches = branches, slice = item.slice,
                center = center_index, method = "sage_no_certificate", radius = item.radius,
                samples = backend_samples, certificate_status = certificate.status,
                competitor_pruning_rate = 0.0,
                global_searches = unpruned.global_searches,
                reference_branch_solves = unpruned.reference_branch_solves,
                competitor_branch_solves = unpruned.competitor_branch_solves,
                unsafe_pruning = unpruned.unsafe_pruning,
                logically_feasible = unpruned.feasible,
                gradient_parity_error = unpruned_parity, seconds = unpruned.seconds,
            ))

            full = full_sage_diagnostics(
                model,
                geometry,
                center_result,
                item.center,
                item.radius,
                target,
                samples,
            )
            full_parity = norm(full.estimate.gradient .- sage_expected)
            maximum_backend_gradient_parity_error = max(
                maximum_backend_gradient_parity_error,
                full_parity,
            )
            push!(workload_rows, (
                dimension = d, branches = branches, slice = item.slice,
                center = center_index, method = "sage_full", radius = item.radius,
                samples = backend_samples, certificate_status = certificate.status,
                competitor_pruning_rate = pruning_rate,
                global_searches = full.estimate.perturbed_global_searches,
                reference_branch_solves = full.estimate.reference_branch_solves,
                competitor_branch_solves = full.estimate.competitor_branch_solves,
                unsafe_pruning = full.unsafe,
                logically_feasible = full.feasible,
                gradient_parity_error = full_parity, seconds = full.seconds,
            ))
        end
    end

    fidelity_columns = (
        :dimension, :branches, :method, :radius, :repetitions, :samples,
        :reference_samples, :relative_bias, :trace_variance, :rmse,
        :mean_cosine, :minimum_cosine,
    )
    workload_columns = (
        :dimension, :branches, :slice, :center, :method, :radius, :samples,
        :certificate_status, :competitor_pruning_rate, :global_searches,
        :reference_branch_solves, :competitor_branch_solves, :unsafe_pruning,
        :logically_feasible, :gradient_parity_error, :seconds,
    )
    fidelity_path = joinpath(output_directory, "phase7_fidelity.csv")
    workload_path = joinpath(output_directory, "phase7_workload.csv")
    write_csv(fidelity_path, fidelity_columns, fidelity_rows)
    write_csv(workload_path, workload_columns, workload_rows)

    sage_fidelity = filter(row -> row.method == "sage_full", fidelity_rows)
    blackbox_fidelity = filter(row -> row.method == "blackbox", fidelity_rows)
    fidelity_gate = all(
        row.mean_cosine >= 0.90 && row.relative_bias <= 0.20
        for row in sage_fidelity
    )
    rmse_gate = all(begin
        blackbox = only(filter(
            candidate -> candidate.dimension == row.dimension &&
                candidate.branches == row.branches,
            blackbox_fidelity,
        ))
        row.rmse <= 1.10 * blackbox.rmse
    end for row in sage_fidelity)

    large_branch_threshold = any(>=(8), branch_counts) ? 8 : minimum(branch_counts)
    comparable_blackbox = filter(
        row -> row.method == "blackbox" && row.branches >= large_branch_threshold,
        workload_rows,
    )
    comparable_sage = filter(
        row -> row.method == "sage_full" && row.branches >= large_branch_threshold,
        workload_rows,
    )
    blackbox_global_searches = sum(row.global_searches for row in comparable_blackbox)
    sage_global_searches = sum(row.global_searches for row in comparable_sage)
    global_search_reduction = 1.0 - sage_global_searches / blackbox_global_searches
    blackbox_seconds = sum(row.seconds for row in comparable_blackbox)
    sage_seconds = sum(row.seconds for row in comparable_sage)
    wall_time_ratio = sage_seconds / blackbox_seconds
    unsafe_pruning = sum(row.unsafe_pruning for row in workload_rows)
    logical_gate = all(row.logically_feasible for row in workload_rows)
    parity_gate = maximum_projection_parity_error < 1e-6 &&
        maximum_backend_gradient_parity_error < 1e-5
    pruning_gate = unsafe_pruning == 0
    global_reduction_gate = global_search_reduction >= 0.25
    wall_time_gate = wall_time_ratio <= 1.0
    phase7_pass = fidelity_gate && rmse_gate && logical_gate && parity_gate &&
        pruning_gate && global_reduction_gate && wall_time_gate

    summary = Dict{String, Any}(
        "phase" => "phase7_scaling_smoke",
        "julia_version" => string(VERSION),
        "dimensions" => dimensions,
        "branch_counts" => branch_counts,
        "large_branch_threshold" => large_branch_threshold,
        "repetitions" => repetitions,
        "estimate_samples" => estimate_samples,
        "reference_samples" => reference_samples,
        "backend_samples" => backend_samples,
        "sampling_scheme" => string(sampling_scheme),
        "maximum_projection_parity_error" => maximum_projection_parity_error,
        "maximum_backend_gradient_parity_error" => maximum_backend_gradient_parity_error,
        "unsafe_pruning" => unsafe_pruning,
        "global_search_reduction" => global_search_reduction,
        "wall_time_ratio" => wall_time_ratio,
        "fidelity_gate" => fidelity_gate,
        "rmse_gate" => rmse_gate,
        "logical_gate" => logical_gate,
        "parity_gate" => parity_gate,
        "pruning_gate" => pruning_gate,
        "global_reduction_gate" => global_reduction_gate,
        "wall_time_gate" => wall_time_gate,
        "pass" => phase7_pass,
    )
    summary_path = joinpath(output_directory, "phase7_summary.json")
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE Phase 7 scaling smoke benchmark")
    println("Julia version: ", VERSION)
    println("Proposal contract: exact feasible forward, compact q beta=2, fixed rho per slice, Eq.(17)-(18)")
    println("dimensions: ", dimensions)
    println("branch counts: ", branch_counts)
    println("repetitions/estimate samples/reference samples/backend samples: ",
        repetitions, "/", estimate_samples, "/", reference_samples, "/", backend_samples)
    println("sampling scheme: ", sampling_scheme)
    println()
    println("configuration\tSAGE relative bias\tSAGE RMSE\tSAGE mean cosine\tblack-box RMSE")
    for row in sage_fidelity
        blackbox = only(filter(
            candidate -> candidate.dimension == row.dimension &&
                candidate.branches == row.branches,
            blackbox_fidelity,
        ))
        println("d=$(row.dimension),K=$(row.branches)\t$(row.relative_bias)\t$(row.rmse)\t$(row.mean_cosine)\t$(blackbox.rmse)")
    end
    println()
    println("maximum projection parity error: ", maximum_projection_parity_error)
    println("maximum backend gradient parity error: ", maximum_backend_gradient_parity_error)
    println("unsafe pruning count: ", unsafe_pruning)
    println("logical feasibility gate: ", logical_gate ? "PASS" : "FAIL")
    println("large-case black-box/SAGE global searches: ",
        blackbox_global_searches, "/", sage_global_searches)
    println("large-case global-search reduction: ", global_search_reduction)
    println("large-case black-box/SAGE seconds: ", blackbox_seconds, "/", sage_seconds)
    println("large-case SAGE/black-box wall-time ratio: ", wall_time_ratio)
    println("fidelity gate: ", fidelity_gate ? "PASS" : "FAIL")
    println("RMSE gate: ", rmse_gate ? "PASS" : "FAIL")
    println("parity gate: ", parity_gate ? "PASS" : "FAIL")
    println("safe-pruning gate: ", pruning_gate ? "PASS" : "FAIL")
    println("global-reduction gate: ", global_reduction_gate ? "PASS" : "FAIL")
    println("wall-time gate: ", wall_time_gate ? "PASS" : "FAIL")
    println("fidelity CSV: ", fidelity_path)
    println("workload CSV: ", workload_path)
    println("summary JSON: ", summary_path)
    println("PHASE7_SCALING_SMOKE=", phase7_pass ? "PASS" : "FAIL")
end

main()
