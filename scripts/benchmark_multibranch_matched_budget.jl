using DisjunctiveNet
using LinearAlgebra
using Random
using Statistics

# Phase 6: proposal E1-E3 controlled multi-branch, matched-budget benchmark.
# The analytic box oracle makes the high-sample numerical reference affordable;
# explicit parity checks tie it back to the certified DisjunctiveNet backend.

struct BoxUnion
    centers::Matrix{Float64} # dimension x branches
    halfwidth::Float64
end

dimension(geometry::BoxUnion) = size(geometry.centers, 1)
branch_count(geometry::BoxUnion) = size(geometry.centers, 2)

function grid_four_boxes(; offset = 0.8, halfwidth = 0.3)
    centers = [
        -offset -offset offset offset
        -offset offset -offset offset
    ]
    return BoxUnion(centers, halfwidth)
end

function ring_boxes(count; radius = 1.5, halfwidth = 0.3)
    centers = zeros(2, count)
    for branch in 1:count
        angle = 2pi * (branch - 1) / count
        centers[:, branch] .= radius .* [cos(angle), sin(angle)]
    end
    return BoxUnion(centers, halfwidth)
end

function disjunctive_model(geometry::BoxUnion)
    d = dimension(geometry)
    model = DisjunctiveModel(d)
    set_bounds!(model, lower = fill(-3.0, d), upper = fill(3.0, d))
    disjuncts = Vector{LinearConstraint}[]
    for branch in 1:branch_count(geometry)
        constraints = LinearConstraint[]
        for coordinate in 1:d
            normal = zeros(d)
            normal[coordinate] = 1.0
            center = geometry.centers[coordinate, branch]
            push!(constraints, LinearConstraint(normal, :>=, center - geometry.halfwidth))
            push!(constraints, LinearConstraint(normal, :<=, center + geometry.halfwidth))
        end
        push!(disjuncts, constraints)
    end
    add_disjunction!(model, disjuncts...)
    return model
end

function project_box(geometry::BoxUnion, point, branch)
    center = view(geometry.centers, :, branch)
    return clamp.(point, center .- geometry.halfwidth, center .+ geometry.halfwidth)
end

function analytic_project(geometry::BoxUnion, point; tie_tol = 1e-12)
    best_branch = 1
    best_y = project_box(geometry, point, 1)
    best_objective = 0.5 * sum(abs2, best_y .- point)
    for branch in 2:branch_count(geometry)
        candidate_y = project_box(geometry, point, branch)
        candidate_objective = 0.5 * sum(abs2, candidate_y .- point)
        if candidate_objective < best_objective - tie_tol
            best_branch = branch
            best_y = candidate_y
            best_objective = candidate_objective
        end
    end
    return (y = collect(best_y), branch = best_branch, objective = best_objective)
end

function box_projection_jacobian(geometry::BoxUnion, point, branch)
    center = view(geometry.centers, :, branch)
    diagonal = [
        center[index] - geometry.halfwidth < point[index] <
        center[index] + geometry.halfwidth ? 1.0 : 0.0
        for index in eachindex(point)
    ]
    return Diagonal(diagonal)
end

task_loss(y, target) = sum(abs2, y .- target)
task_loss_gradient(y, target) = 2.0 .* (y .- target)

function sample_matrix(rng, d, count; beta = 2.0)
    return reduce(hcat, [sample_ball_smoothing(rng, d; beta = beta) for _ in 1:count])
end

function analytic_sage_gradient(geometry, z, radius, target, samples; beta = 2.0)
    center = analytic_project(geometry, z)
    total = zeros(length(z))
    switching = 0
    for column in axes(samples, 2)
        u = samples[:, column]
        w = z .+ radius .* u
        local_y = project_box(geometry, w, center.branch)
        local_gradient = box_projection_jacobian(geometry, w, center.branch)' *
            task_loss_gradient(local_y, target)
        global_result = analytic_project(geometry, w)
        residual = task_loss(global_result.y, target) - task_loss(local_y, target)
        total .+= local_gradient .+
            (residual / radius) .* ball_smoothing_score(u; beta = beta)
        switching += global_result.branch != center.branch
    end
    return (gradient = total ./ size(samples, 2), switching = switching)
end

function analytic_blackbox_gradient(geometry, z, radius, target, samples; beta = 2.0)
    center_loss = task_loss(analytic_project(geometry, z).y, target)
    total = zeros(length(z))
    for column in axes(samples, 2)
        u = samples[:, column]
        global_y = analytic_project(geometry, z .+ radius .* u).y
        residual = task_loss(global_y, target) - center_loss
        total .+= (residual / radius) .* ball_smoothing_score(u; beta = beta)
    end
    return total ./ size(samples, 2)
end

function cosine_similarity(left, right)
    denominator = norm(left) * norm(right)
    return denominator == 0.0 ? NaN : dot(left, right) / denominator
end

function estimator_summary(estimates, reference)
    mean_estimate = reduce(+, estimates) ./ length(estimates)
    bias = mean_estimate .- reference
    trace_variance = mean(sum(abs2, estimate .- mean_estimate) for estimate in estimates)
    rmse = sqrt(mean(sum(abs2, estimate .- reference) for estimate in estimates))
    cosines = [cosine_similarity(estimate, reference) for estimate in estimates]
    return (
        mean = mean_estimate,
        relative_bias = norm(bias) / norm(reference),
        trace_variance = trace_variance,
        rmse = rmse,
        mean_cosine = mean(cosines),
        minimum_cosine = minimum(cosines),
    )
end

function backend_blackbox_gradient(model, z, radius, target, samples; beta = 2.0)
    center = lazy_branch_search(model, z; store_prune_records = false)
    @assert center.status == :optimal
    center_loss = task_loss(center.y, target)
    total = zeros(length(z))
    leaf_solves = 0
    relaxation_solves = 0
    feasible = true
    started = time_ns()
    for column in axes(samples, 2)
        u = samples[:, column]
        result = lazy_branch_search(
            model,
            z .+ radius .* u;
            store_prune_records = false,
        )
        @assert result.status == :optimal
        feasible &= result.report !== nothing && result.report.feasible
        residual = task_loss(result.y, target) - center_loss
        total .+= (residual / radius) .* ball_smoothing_score(u; beta = beta)
        leaf_solves += result.leaf_solves
        relaxation_solves += result.relaxation_solves
    end
    elapsed = (time_ns() - started) / 1e9
    return (
        gradient = total ./ size(samples, 2),
        global_searches = size(samples, 2),
        leaf_solves = leaf_solves,
        relaxation_solves = relaxation_solves,
        feasible = feasible,
        seconds = elapsed,
    )
end

# ---------------------------------------------------------------------------
# Part A: exact-backend parity on a four-branch 2D union.
# ---------------------------------------------------------------------------

function main()
fidelity_geometry = grid_four_boxes()
fidelity_model = disjunctive_model(fidelity_geometry)
fidelity_target = copy(fidelity_geometry.centers[:, 4])
fidelity_z = [-0.45, -0.15]
fidelity_radius = 0.65
parity_samples = sample_matrix(MersenneTwister(20261113), 2, 128)

for point in (
    [-1.2, -1.1], [-0.45, -0.15], [0.0, 0.0], [0.8, 0.8], [1.5, -0.2],
)
    analytic = analytic_project(fidelity_geometry, point)
    backend = lazy_branch_search(fidelity_model, point; store_prune_records = false)
    @assert backend.status == :optimal
    @assert only(backend.choices) == analytic.branch
    @assert isapprox(backend.y, analytic.y; atol = 5e-7)
    @assert backend.report !== nothing && backend.report.feasible
end

backend_center = lazy_branch_search(
    fidelity_model,
    fidelity_z;
    store_prune_records = false,
)
backend_parity = sage_gradient(
    fidelity_model,
    fidelity_z,
    fidelity_radius,
    y -> task_loss(y, fidelity_target),
    y -> task_loss_gradient(y, fidelity_target);
    samples = parity_samples,
    center_result = backend_center,
    store_records = true,
)
analytic_parity = analytic_sage_gradient(
    fidelity_geometry,
    fidelity_z,
    fidelity_radius,
    fidelity_target,
    parity_samples,
)
@assert backend_parity.status == :ok
parity_error = norm(backend_parity.gradient .- analytic_parity.gradient)
@assert parity_error < 1e-6
@assert all(record.status == :ok for record in backend_parity.records)
@assert all(
    check_feasibility(fidelity_model, record.global_y).feasible
    for record in backend_parity.records
)

# ---------------------------------------------------------------------------
# Part B: proposal Eq. (17)-(18) fidelity under matched q, rho, samples, and
# global-query opportunity budget. The reference uses a much larger independent
# proposal-faithful sample, not a different objective.
# ---------------------------------------------------------------------------

reference_samples = sample_matrix(MersenneTwister(20261114), 2, 100_000)
reference = analytic_sage_gradient(
    fidelity_geometry,
    fidelity_z,
    fidelity_radius,
    fidelity_target,
    reference_samples,
).gradient
@assert norm(reference) > 1e-6

repetitions = 30
samples_per_estimate = 64
sage_estimates = Vector{Vector{Float64}}()
blackbox_estimates = Vector{Vector{Float64}}()
for repetition in 1:repetitions
    samples = sample_matrix(
        MersenneTwister(20261114 + repetition),
        2,
        samples_per_estimate,
    )
    push!(sage_estimates, analytic_sage_gradient(
        fidelity_geometry,
        fidelity_z,
        fidelity_radius,
        fidelity_target,
        samples,
    ).gradient)
    push!(blackbox_estimates, analytic_blackbox_gradient(
        fidelity_geometry,
        fidelity_z,
        fidelity_radius,
        fidelity_target,
        samples,
    ))
end

sage_summary = estimator_summary(sage_estimates, reference)
blackbox_summary = estimator_summary(blackbox_estimates, reference)

# ---------------------------------------------------------------------------
# Part C: eight-branch backend workload. Stable centers and adjacent-boundary
# centers share the same fixed radius and N. SAGE pays certificate/local costs;
# black-box pays one global search for every perturbation.
# ---------------------------------------------------------------------------

scaling_geometry = ring_boxes(8)
scaling_model = disjunctive_model(scaling_geometry)
scaling_target = copy(scaling_geometry.centers[:, 5])
scaling_radius = 0.3
workload_samples = 12
stable_centers = [copy(scaling_geometry.centers[:, branch]) for branch in 1:8]
switching_centers = [
    0.5 .* (
        scaling_geometry.centers[:, branch] .+
        scaling_geometry.centers[:, mod1(branch + 1, 8)]
    )
    for branch in 1:8
]
workload_centers = vcat(stable_centers, switching_centers)

# Warm both code paths before timing.
warm_samples = sample_matrix(MersenneTwister(7), 2, 2)
warm_center = stable_centers[1]
warm_result = lazy_branch_search(scaling_model, warm_center; store_prune_records = false)
sage_gradient(
    scaling_model,
    warm_center,
    scaling_radius,
    y -> task_loss(y, scaling_target),
    y -> task_loss_gradient(y, scaling_target);
    samples = warm_samples,
    center_result = warm_result,
    store_records = false,
)
backend_blackbox_gradient(
    scaling_model,
    warm_center,
    scaling_radius,
    scaling_target,
    warm_samples,
)

certificate_count = 0
total_competitors = 0
pruned_competitors = 0
sage_global_searches = 0
sage_reference_solves = 0
sage_competitor_solves = 0
sage_seconds = 0.0
blackbox_global_searches = 0
blackbox_leaf_solves = 0
blackbox_relaxation_solves = 0
blackbox_seconds = 0.0
workload_feasible = true

for (index, center) in enumerate(workload_centers)
    samples = sample_matrix(
        MersenneTwister(20261200 + index),
        2,
        workload_samples,
    )
    center_result = lazy_branch_search(
        scaling_model,
        center;
        store_prune_records = false,
    )
    @assert center_result.status == :optimal
    certificate = switching_certificate(
        scaling_model,
        center,
        scaling_radius;
        center_result = center_result,
    )
    certificate_count += certifies_stability(certificate)
    total_competitors += certificate.total_competitor_branches
    pruned_competitors += certificate.stable_pruned_branches

    started = time_ns()
    sage = sage_gradient(
        scaling_model,
        center,
        scaling_radius,
        y -> task_loss(y, scaling_target),
        y -> task_loss_gradient(y, scaling_target);
        samples = samples,
        center_result = center_result,
        certificate = certificate,
        store_records = true,
    )
    sage_seconds += (time_ns() - started) / 1e9
    @assert sage.status == :ok
    workload_feasible &= all(
        record.status == :ok &&
        check_feasibility(scaling_model, record.global_y).feasible
        for record in sage.records
    )
    sage_global_searches += sage.perturbed_global_searches
    sage_reference_solves += sage.reference_branch_solves
    sage_competitor_solves += sage.competitor_branch_solves

    blackbox = backend_blackbox_gradient(
        scaling_model,
        center,
        scaling_radius,
        scaling_target,
        samples,
    )
    workload_feasible &= blackbox.feasible
    blackbox_global_searches += blackbox.global_searches
    blackbox_leaf_solves += blackbox.leaf_solves
    blackbox_relaxation_solves += blackbox.relaxation_solves
    blackbox_seconds += blackbox.seconds
end

certificate_rate = certificate_count / length(workload_centers)
pruning_rate = pruned_competitors / total_competitors
global_call_reduction = 1.0 - sage_global_searches / blackbox_global_searches
variance_ratio = sage_summary.trace_variance / blackbox_summary.trace_variance
wall_time_ratio = sage_seconds / blackbox_seconds

# Pre-registered Phase 6 gate. Correctness/fidelity are mandatory. The proposal's
# efficiency phase continues only if structure reduces either estimator variance
# or measured total backward time, while also reducing perturbed global searches.
fidelity_pass = parity_error < 1e-6 &&
                sage_summary.mean_cosine >= 0.9 &&
                sage_summary.relative_bias <= 0.2
global_reduction_pass = sage_global_searches < blackbox_global_searches
efficiency_pass = variance_ratio < 1.0 || wall_time_ratio < 1.0
phase6_pass = fidelity_pass && global_reduction_pass && efficiency_pass && workload_feasible

println("SAGE Phase 6 multi-branch matched-budget benchmark")
println("Julia version: ", VERSION)
println("Proposal contract: compact q, beta=2, fixed rho per slice, Eq.(17)-(18), exact feasible forwards")
println()
println("[A] Analytic/backend parity")
println("branches/dimension: 4/2")
println("parity error: ", parity_error)
println("backend switching samples: ", backend_parity.switching_samples, "/", size(parity_samples, 2))
println("backend competitor branch solves: ", backend_parity.competitor_branch_solves)
println()
println("[B] Gradient fidelity, matched q/rho/N and common random numbers")
println("reference gradient: ", reference)
println("repetitions/samples: ", repetitions, "/", samples_per_estimate)
println("method\trelative_bias\ttrace_variance\trmse\tmean_cosine\tminimum_cosine")
println(join((
    "blackbox",
    blackbox_summary.relative_bias,
    blackbox_summary.trace_variance,
    blackbox_summary.rmse,
    blackbox_summary.mean_cosine,
    blackbox_summary.minimum_cosine,
), '\t'))
println(join((
    "sage",
    sage_summary.relative_bias,
    sage_summary.trace_variance,
    sage_summary.rmse,
    sage_summary.mean_cosine,
    sage_summary.minimum_cosine,
), '\t'))
println("SAGE/blackbox variance ratio: ", variance_ratio)
println()
println("[C] Eight-branch certified workload, same centers/rho/N")
println("workload centers/stable candidates/switching candidates: ", length(workload_centers), "/8/8")
println("samples per center: ", workload_samples)
println("certificate rate: ", certificate_rate)
println("competitor pruning rate: ", pruning_rate)
println("blackbox global searches: ", blackbox_global_searches)
println("SAGE global searches: ", sage_global_searches)
println("global-search reduction: ", global_call_reduction)
println("SAGE reference/competitor branch solves: ", sage_reference_solves, "/", sage_competitor_solves)
println("blackbox leaf/relaxation solves: ", blackbox_leaf_solves, "/", blackbox_relaxation_solves)
println("blackbox/SAGE seconds: ", blackbox_seconds, "/", sage_seconds)
println("SAGE/blackbox wall-time ratio: ", wall_time_ratio)
println("all perturbed outputs logically feasible: ", workload_feasible)
println()
println("fidelity gate: ", fidelity_pass ? "PASS" : "FAIL")
println("global-search reduction gate: ", global_reduction_pass ? "PASS" : "FAIL")
println("variance-or-wall-time efficiency gate: ", efficiency_pass ? "PASS" : "FAIL")
println("PHASE6_MATCHED_BUDGET=", phase6_pass ? "PASS" : "FAIL")
end

main()

