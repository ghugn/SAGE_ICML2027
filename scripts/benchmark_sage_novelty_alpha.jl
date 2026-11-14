using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

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
        entries = [json_value(string(key)) * ":" * json_value(item)
                   for (key, item) in value]
        return "{" * join(entries, ',') * "}"
    end
    error("Unsupported JSON value $(typeof(value)).")
end

function interval_union_model()
    model = DisjunctiveModel(1)
    set_bounds!(model, lower = [-2.0], upper = [2.0])
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, -1.0)],
        [LinearConstraint([1.0], :>=, 1.0)],
    )
    return model
end

const TARGET = 1.5
loss_scalar(y) = (y - TARGET)^2
left_projection(z) = clamp(z, -2.0, -1.0)
right_projection(z) = clamp(z, 1.0, 2.0)

function winner(z)
    left_distance = abs(z - left_projection(z))
    right_distance = abs(z - right_projection(z))
    return left_distance <= right_distance ? 1 : 2
end

branch_projection(z, branch) = branch == 1 ? left_projection(z) : right_projection(z)
global_projection(z) = branch_projection(z, winner(z))

function branch_jacobian(z, branch)
    lower, upper = branch == 1 ? (-2.0, -1.0) : (1.0, 2.0)
    return lower < z < upper ? 1.0 : 0.0
end

"""Proposal equation (18) integrand for one compact-q perturbation."""
function sage_integrand(z, radius, u; beta = 2.0)
    reference = winner(z)
    w = z + radius * u
    local_y = branch_projection(w, reference)
    local_gradient = branch_jacobian(w, reference) * 2.0 * (local_y - TARGET)
    residual = loss_scalar(global_projection(w)) - loss_scalar(local_y)
    score = only(ball_smoothing_score([u]; beta = beta))
    return local_gradient + (residual / radius) * score
end

function analytic_reference(z, radius)
    # On the registered slice, every perturbation remains in the gap and the
    # switch is u > -z/rho. Integrating q(u)=15/16(1-u^2)^2 gives this value.
    threshold = -z / radius
    0.0 < threshold < 1.0 || error("Registered reference requires 0 < -z/rho < 1.")
    return -11.25 * (1.0 - threshold^2)^2
end

function sample_q_matrix(rng, count; antithetic = false)
    if antithetic
        return sample_antithetic_ball_smoothing(rng, 1, count; beta = 2.0)
    end
    return reshape(
        [only(sample_ball_smoothing(rng, 1; beta = 2.0)) for _ in 1:count],
        1,
        :,
    )
end

function estimate_from_samples(z, radius, samples, weights = nothing)
    n = size(samples, 2)
    importance = weights === nothing ? ones(n) : weights
    return sum(
        importance[index] * sage_integrand(z, radius, samples[1, index])
        for index in 1:n
    ) / n
end

function scalar_summary(values, reference)
    average = mean(values)
    variance = var(values; corrected = true)
    rmse = sqrt(mean((value - reference)^2 for value in values))
    return (
        mean = average,
        bias = average - reference,
        relative_bias = abs(average - reference) / abs(reference),
        variance = variance,
        rmse = rmse,
        sign_match = mean(sign(value) == sign(reference) for value in values),
    )
end

function exact_record_check(model, record)
    exact_y = global_projection(only(record.w))
    exact_branch = winner(only(record.w))
    return record.status == :ok &&
        only(record.global_choices) == exact_branch &&
        isapprox(only(record.global_y), exact_y; atol = 5e-7) &&
        is_logically_feasible(model, record.global_y)
end

function main()
    repetitions = parse(Int, get(ENV, "SAGE_ALPHA_REPETITIONS", "100"))
    samples = parse(Int, get(ENV, "SAGE_ALPHA_SAMPLES", "64"))
    moment_samples = parse(Int, get(ENV, "SAGE_ALPHA_MOMENT_SAMPLES", "50000"))
    backend_repetitions = parse(Int, get(ENV, "SAGE_ALPHA_BACKEND_REPETITIONS", "5"))
    backend_samples = parse(Int, get(ENV, "SAGE_ALPHA_BACKEND_SAMPLES", "64"))
    output_directory = abspath(get(
        ENV, "SAGE_ALPHA_OUTPUT", "sage_alpha_novelty_results",
    ))
    repetitions >= 30 || error("At least 30 fidelity repetitions are required.")
    samples > 0 && iseven(samples) || error("Sample count must be positive and even.")
    moment_samples >= 20_000 || error("At least 20,000 moment samples are required.")
    backend_repetitions >= 3 || error("At least three backend repetitions are required.")
    backend_samples > 0 || error("Backend sample count must be positive.")
    mkpath(output_directory)

    beta = 2.0
    radius = 0.5
    mixture_weight = 0.5
    focus_strength = 0.9
    normal = reshape([1.0], 1, 1)
    centers = [-0.3, -0.2, -0.1]

    # Gate A: the proposal must preserve expectations under q before any
    # gradient or efficiency result is considered.
    moment_normals = [1.0 0.0; 0.0 1.0]
    moment_batch = sample_geometry_smoothing(
        MersenneTwister(20261114001),
        2,
        moment_samples,
        moment_normals;
        beta = beta,
        mixture_weight = mixture_weight,
        focus_strength = focus_strength,
    )
    weighted_mass = mean(moment_batch.importance_weights)
    weighted_mean = moment_batch.samples * moment_batch.importance_weights ./ moment_samples
    expected_second = 1.0 / (2 + 2beta + 2)
    weighted_seconds = [
        mean(moment_batch.importance_weights .* abs2.(moment_batch.samples[row, :]))
        for row in 1:2
    ]
    mass_error = abs(weighted_mass - 1.0)
    mean_error = norm(weighted_mean)
    second_error = maximum(abs.(weighted_seconds .- expected_second))
    density_gate = mass_error <= 0.02 && mean_error <= 0.02 && second_error <= 0.015

    # Gate B: controlled analytical switching slices, equal N and the same
    # compact-q target. Antithetic q is retained as the strongest existing
    # isotropic control variate rather than comparing only to naive sampling.
    fidelity_rows = NamedTuple[]
    per_slice_geometry_independent_ratios = Float64[]
    per_slice_geometry_antithetic_ratios = Float64[]
    geometry_relative_biases = Float64[]
    total_independent_variance = 0.0
    total_antithetic_variance = 0.0
    total_geometry_variance = 0.0
    total_independent_mse = 0.0
    total_antithetic_mse = 0.0
    total_geometry_mse = 0.0

    for (slice_index, z) in enumerate(centers)
        reference = analytic_reference(z, radius)
        estimates = Dict(
            :isotropic => Float64[],
            :antithetic => Float64[],
            :geometry => Float64[],
        )
        for repetition in 1:repetitions
            seed = 20261114010 + 1000slice_index + repetition
            independent = sample_q_matrix(MersenneTwister(seed), samples)
            antithetic = sample_q_matrix(MersenneTwister(seed), samples; antithetic = true)
            geometry = sample_geometry_smoothing(
                MersenneTwister(seed),
                1,
                samples,
                normal;
                beta = beta,
                mixture_weight = mixture_weight,
                focus_strength = focus_strength,
            )
            push!(estimates[:isotropic], estimate_from_samples(z, radius, independent))
            push!(estimates[:antithetic], estimate_from_samples(z, radius, antithetic))
            push!(estimates[:geometry], estimate_from_samples(
                z,
                radius,
                geometry.samples,
                geometry.importance_weights,
            ))
        end
        summaries = Dict(method => scalar_summary(values, reference)
                         for (method, values) in estimates)
        for method in (:isotropic, :antithetic, :geometry)
            summary = summaries[method]
            push!(fidelity_rows, (
                center = z,
                method = method,
                repetitions = repetitions,
                samples = samples,
                reference = reference,
                mean = summary.mean,
                bias = summary.bias,
                relative_bias = summary.relative_bias,
                variance = summary.variance,
                rmse = summary.rmse,
                sign_match = summary.sign_match,
            ))
        end
        geometry_independent_ratio = summaries[:geometry].variance / summaries[:isotropic].variance
        geometry_antithetic_ratio = summaries[:geometry].variance / summaries[:antithetic].variance
        push!(per_slice_geometry_independent_ratios, geometry_independent_ratio)
        push!(per_slice_geometry_antithetic_ratios, geometry_antithetic_ratio)
        push!(geometry_relative_biases, summaries[:geometry].relative_bias)
        total_independent_variance += summaries[:isotropic].variance
        total_antithetic_variance += summaries[:antithetic].variance
        total_geometry_variance += summaries[:geometry].variance
        total_independent_mse += summaries[:isotropic].rmse^2
        total_antithetic_mse += summaries[:antithetic].rmse^2
        total_geometry_mse += summaries[:geometry].rmse^2
    end

    geometry_independent_variance_ratio = total_geometry_variance / total_independent_variance
    geometry_antithetic_variance_ratio = total_geometry_variance / total_antithetic_variance
    geometry_independent_rmse_ratio = sqrt(total_geometry_mse / total_independent_mse)
    geometry_antithetic_rmse_ratio = sqrt(total_geometry_mse / total_antithetic_mse)
    fidelity_gate = maximum(geometry_relative_biases) <= 0.10
    variance_gate = geometry_independent_variance_ratio <= 0.75 &&
        geometry_antithetic_variance_ratio <= 0.85 &&
        maximum(per_slice_geometry_independent_ratios) <= 0.90

    # Gate C: backend parity, feasibility, certificate ablation, and complete
    # geometry overhead. Base and Cert use identical q samples per repetition.
    model = interval_union_model()
    backend_z = [-0.1]
    loss(y) = loss_scalar(only(y))
    loss_gradient(y) = [2.0 * (only(y) - TARGET)]
    backend_rows = NamedTuple[]
    maximum_base_cert_parity = 0.0
    maximum_geometry_direct_parity = 0.0
    unsafe_outputs = 0
    base_global_searches = 0
    certified_global_searches = 0
    geometry_global_searches = 0
    base_total_branch_solves = 0
    certified_total_branch_solves = 0
    geometry_total_branch_solves = 0
    geometry_effective_samples = Float64[]

    for repetition in 1:backend_repetitions
        seed = 20261114100 + repetition
        center_result = lazy_branch_search(model, backend_z; store_prune_records = false)
        base = sage_gradient(
            model,
            backend_z,
            radius,
            loss,
            loss_gradient;
            n_samples = backend_samples,
            rng = MersenneTwister(seed),
            use_ball_certificate = false,
            use_sample_certificate = false,
            store_records = true,
        )
        certified = sage_gradient(
            model,
            backend_z,
            radius,
            loss,
            loss_gradient;
            n_samples = backend_samples,
            rng = MersenneTwister(seed),
            center_result = center_result,
            store_records = true,
        )
        geometry = sage_gradient(
            model,
            backend_z,
            radius,
            loss,
            loss_gradient;
            n_samples = backend_samples,
            rng = MersenneTwister(seed),
            center_result = center_result,
            sampling = :geometry,
            geometry_mixture_weight = mixture_weight,
            geometry_focus_strength = focus_strength,
            store_records = true,
        )
        all(estimate.status == :ok for estimate in (base, certified, geometry)) ||
            error("A backend novelty ablation was unresolved.")

        base_cert_parity = norm(base.gradient .- certified.gradient)
        geometry_direct = sum(
            record.importance_weight * sage_integrand(
                only(backend_z), radius, only(record.u),
            ) for record in geometry.records
        ) / geometry.samples
        geometry_direct_parity = abs(only(geometry.gradient) - geometry_direct)
        maximum_base_cert_parity = max(maximum_base_cert_parity, base_cert_parity)
        maximum_geometry_direct_parity = max(
            maximum_geometry_direct_parity,
            geometry_direct_parity,
        )
        unsafe_outputs += count(!exact_record_check(model, record)
                                for estimate in (base, certified, geometry)
                                for record in estimate.records)

        base_calls = base.reference_branch_solves + base.competitor_branch_solves +
            base.geometry_branch_solves
        certified_calls = certified.reference_branch_solves +
            certified.competitor_branch_solves + certified.geometry_branch_solves
        geometry_calls = geometry.reference_branch_solves +
            geometry.competitor_branch_solves + geometry.geometry_branch_solves
        base_global_searches += base.perturbed_global_searches
        certified_global_searches += certified.perturbed_global_searches
        geometry_global_searches += geometry.perturbed_global_searches
        base_total_branch_solves += base_calls
        certified_total_branch_solves += certified_calls
        geometry_total_branch_solves += geometry_calls
        push!(geometry_effective_samples, geometry.importance_effective_sample_size)

        for (method, estimate, calls) in (
            (:sage_base, base, base_calls),
            (:sage_cert, certified, certified_calls),
            (:sage_geometry, geometry, geometry_calls),
        )
            push!(backend_rows, (
                repetition = repetition,
                method = method,
                gradient = only(estimate.gradient),
                global_searches = estimate.perturbed_global_searches,
                reference_branch_solves = estimate.reference_branch_solves,
                competitor_branch_solves = estimate.competitor_branch_solves,
                geometry_branch_solves = estimate.geometry_branch_solves,
                total_branch_solves = calls,
                sample_certificates = estimate.sample_stability_certificates,
                switching_samples = estimate.switching_samples,
                effective_sample_size = estimate.importance_effective_sample_size,
                seconds = estimate.solve_time,
            ))
        end
    end

    global_search_reduction = 1.0 - geometry_global_searches / base_global_searches
    branch_solve_reduction = 1.0 - geometry_total_branch_solves / base_total_branch_solves
    parity_gate = maximum_base_cert_parity <= 1e-8 &&
        maximum_geometry_direct_parity <= 1e-8
    safe_feasibility_gate = unsafe_outputs == 0
    compute_gate = global_search_reduction >= 0.25 && branch_solve_reduction >= 0.15
    alpha_pass = density_gate && fidelity_gate && variance_gate && parity_gate &&
        safe_feasibility_gate && compute_gate

    fidelity_path = joinpath(output_directory, "sage_alpha_fidelity.csv")
    backend_path = joinpath(output_directory, "sage_alpha_backend.csv")
    summary_path = joinpath(output_directory, "sage_alpha_summary.json")
    write_csv(
        fidelity_path,
        (
            :center, :method, :repetitions, :samples, :reference, :mean, :bias,
            :relative_bias, :variance, :rmse, :sign_match,
        ),
        fidelity_rows,
    )
    write_csv(
        backend_path,
        (
            :repetition, :method, :gradient, :global_searches,
            :reference_branch_solves, :competitor_branch_solves,
            :geometry_branch_solves, :total_branch_solves, :sample_certificates,
            :switching_samples, :effective_sample_size, :seconds,
        ),
        backend_rows,
    )
    summary = Dict(
        "status" => alpha_pass ? "PASS" : "FAIL",
        "density_gate" => density_gate,
        "fidelity_gate" => fidelity_gate,
        "variance_gate" => variance_gate,
        "parity_gate" => parity_gate,
        "safe_feasibility_gate" => safe_feasibility_gate,
        "compute_gate" => compute_gate,
        "moment_mass_error" => mass_error,
        "moment_mean_error" => mean_error,
        "moment_second_error" => second_error,
        "geometry_independent_variance_ratio" => geometry_independent_variance_ratio,
        "geometry_antithetic_variance_ratio" => geometry_antithetic_variance_ratio,
        "geometry_independent_rmse_ratio" => geometry_independent_rmse_ratio,
        "geometry_antithetic_rmse_ratio" => geometry_antithetic_rmse_ratio,
        "per_slice_geometry_independent_variance_ratios" => per_slice_geometry_independent_ratios,
        "per_slice_geometry_antithetic_variance_ratios" => per_slice_geometry_antithetic_ratios,
        "maximum_geometry_relative_bias" => maximum(geometry_relative_biases),
        "maximum_base_cert_parity" => maximum_base_cert_parity,
        "maximum_geometry_direct_parity" => maximum_geometry_direct_parity,
        "unsafe_outputs" => unsafe_outputs,
        "base_global_searches" => base_global_searches,
        "certified_global_searches" => certified_global_searches,
        "geometry_global_searches" => geometry_global_searches,
        "global_search_reduction" => global_search_reduction,
        "base_total_branch_solves" => base_total_branch_solves,
        "certified_total_branch_solves" => certified_total_branch_solves,
        "geometry_total_branch_solves" => geometry_total_branch_solves,
        "branch_solve_reduction" => branch_solve_reduction,
        "mean_geometry_effective_sample_size" => mean(geometry_effective_samples),
        "configuration" => Dict(
            "repetitions" => repetitions,
            "samples" => samples,
            "backend_repetitions" => backend_repetitions,
            "backend_samples" => backend_samples,
            "beta" => beta,
            "radius" => radius,
            "mixture_weight" => mixture_weight,
            "focus_strength" => focus_strength,
        ),
    )
    open(summary_path, "w") do io
        println(io, json_value(summary))
    end

    println("SAGE research-alpha novelty gate")
    println("Julia version: ", VERSION)
    println("Contract: Eq.(17)-(18) target q; Eq.(19)-(21) normals; exact q/p importance weights")
    println("repetitions/samples/backend-repetitions/backend-samples: ",
        repetitions, "/", samples, "/", backend_repetitions, "/", backend_samples)
    println()
    println("[A] Proposal-density preservation")
    println("mass/mean/second-moment error: ", mass_error, "/", mean_error, "/", second_error)
    println("density gate: ", density_gate ? "PASS" : "FAIL")
    println()
    println("[B] Matched-N analytical switching fidelity")
    println("center\tmethod\trelative_bias\tvariance\trmse")
    for row in fidelity_rows
        println(row.center, '\t', row.method, '\t', row.relative_bias, '\t', row.variance, '\t', row.rmse)
    end
    println("geometry/isotropic aggregate variance ratio: ", geometry_independent_variance_ratio)
    println("geometry/antithetic aggregate variance ratio: ", geometry_antithetic_variance_ratio)
    println("geometry/isotropic aggregate RMSE ratio: ", geometry_independent_rmse_ratio)
    println("geometry/antithetic aggregate RMSE ratio: ", geometry_antithetic_rmse_ratio)
    println("fidelity gate: ", fidelity_gate ? "PASS" : "FAIL")
    println("variance gate: ", variance_gate ? "PASS" : "FAIL")
    println()
    println("[C] Certified backend ablation")
    println("maximum base/cert gradient parity error: ", maximum_base_cert_parity)
    println("maximum geometry/backend direct parity error: ", maximum_geometry_direct_parity)
    println("unsafe or infeasible outputs: ", unsafe_outputs)
    println("base/cert/geometry global searches: ",
        base_global_searches, "/", certified_global_searches, "/", geometry_global_searches)
    println("global-search reduction vs SAGE-Base: ", global_search_reduction)
    println("base/cert/geometry complete branch solves: ",
        base_total_branch_solves, "/", certified_total_branch_solves, "/", geometry_total_branch_solves)
    println("branch-solve reduction vs SAGE-Base: ", branch_solve_reduction)
    println("mean geometry effective sample size: ", mean(geometry_effective_samples), "/", backend_samples)
    println("parity gate: ", parity_gate ? "PASS" : "FAIL")
    println("safe-feasibility gate: ", safe_feasibility_gate ? "PASS" : "FAIL")
    println("compute gate: ", compute_gate ? "PASS" : "FAIL")
    println("fidelity CSV: ", fidelity_path)
    println("backend CSV: ", backend_path)
    println("summary JSON: ", summary_path)
    println("SAGE_ALPHA_NOVELTY=", alpha_pass ? "PASS" : "FAIL")
    alpha_pass || exit(1)
end

main()
