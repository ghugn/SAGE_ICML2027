using DisjunctiveNet
using Random
using Statistics

# Controlled go/no-go benchmark for learning through a branch discontinuity.
# The scalar closed form keeps the training comparison cheap and transparent;
# parity checks below connect it to the certified DisjunctiveNet backend.

const TARGET = 1.5
const INITIAL_Z = -0.4
const RADIUS = 0.5
const LEARNING_RATE = 0.03
const STEPS = 30
const SAMPLES_PER_STEP = 128
const SEEDS = 10
const METHODS = (:hard_local, :ste, :blackbox, :sage)

loss(y) = (y - TARGET)^2
loss_gradient(y) = 2.0 * (y - TARGET)

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

function draw_samples(rng, count)
    return [only(sample_ball_smoothing(rng, 1; beta = 2.0)) for _ in 1:count]
end

function hard_local_gradient(z)
    branch = winner(z)
    y = branch_projection(z, branch)
    return branch_jacobian(z, branch) * loss_gradient(y)
end

ste_gradient(z) = loss_gradient(global_projection(z))

function blackbox_gradient(z, samples)
    center_loss = loss(global_projection(z))
    total = 0.0
    for u in samples
        perturbed_loss = loss(global_projection(z + RADIUS * u))
        score = only(ball_smoothing_score([u]; beta = 2.0))
        total += ((perturbed_loss - center_loss) / RADIUS) * score
    end
    return total / length(samples)
end

function direct_sage_gradient(z, samples)
    reference_branch = winner(z)
    total = 0.0
    for u in samples
        w = z + RADIUS * u
        local_y = branch_projection(w, reference_branch)
        local_gradient = branch_jacobian(w, reference_branch) * loss_gradient(local_y)
        residual = loss(global_projection(w)) - loss(local_y)
        score = only(ball_smoothing_score([u]; beta = 2.0))
        total += local_gradient + (residual / RADIUS) * score
    end
    return total / length(samples)
end

function method_gradient(method, z, samples)
    method == :hard_local && return hard_local_gradient(z)
    method == :ste && return ste_gradient(z)
    method == :blackbox && return blackbox_gradient(z, samples)
    method == :sage && return direct_sage_gradient(z, samples)
    error("Unknown method: $method")
end

function train(method, seed)
    rng = MersenneTwister(20261113 + seed)
    z = INITIAL_Z
    trajectory = Float64[z]
    solver_calls = 0
    for _ in 1:STEPS
        samples = draw_samples(rng, SAMPLES_PER_STEP)
        gradient = method_gradient(method, z, samples)
        # Common bounded-domain safeguard; it is identical for all methods.
        z = clamp(z - LEARNING_RATE * gradient, -3.0, 3.0)
        push!(trajectory, z)
        solver_calls += method in (:blackbox, :sage) ? SAMPLES_PER_STEP : 1
    end
    final_y = global_projection(z)
    crossed = any(value -> value > 0.0, trajectory)
    first_crossing = findfirst(value -> value > 0.0, trajectory)
    return (
        method = method,
        seed = seed,
        final_z = z,
        final_y = final_y,
        final_loss = loss(final_y),
        crossed = crossed,
        first_crossing = isnothing(first_crossing) ? -1 : first_crossing - 1,
        solver_calls = solver_calls,
        trajectory = trajectory,
    )
end

model = interval_union_model()

# Backend parity: the closed form must agree with certified lazy search.
for z in range(-3.0, 3.0; length = 61)
    result = lazy_branch_search(model, [z])
    @assert result.status == :optimal
    # HiGHS' interior-point solution may sit a few 1e-7 from an inactive box
    # coordinate even though the accepted logical result is exact to tolerance.
    @assert isapprox(only(result.y), global_projection(z); atol = 5e-7)
    @assert only(result.choices) == winner(z)
end

# Estimator parity using exactly the same perturbations.
parity_samples = draw_samples(MersenneTwister(20261113), 256)
backend_estimate = sage_gradient(
    model,
    [INITIAL_Z],
    RADIUS,
    y -> loss(only(y)),
    y -> [loss_gradient(only(y))];
    samples = reshape(parity_samples, 1, :),
    store_records = false,
)
@assert backend_estimate.status == :ok
direct_estimate = direct_sage_gradient(INITIAL_Z, parity_samples)
@assert isapprox(only(backend_estimate.gradient), direct_estimate; atol = 1e-7)

results = [train(method, seed) for method in METHODS for seed in 1:SEEDS]

println("SAGE discontinuity-learning go/no-go benchmark")
println("Julia version: ", VERSION)
println("Configuration: z0=$INITIAL_Z target=$TARGET rho=$RADIUS lr=$LEARNING_RATE steps=$STEPS samples=$SAMPLES_PER_STEP seeds=$SEEDS")
println("Backend/direct SAGE parity error: ", abs(only(backend_estimate.gradient) - direct_estimate))
println()
println("method\tswitch_rate\tmedian_crossing_step\tmean_final_loss\tmedian_final_loss\tmean_solver_calls")

summaries = Dict{Symbol, NamedTuple}()
for method in METHODS
    group = filter(result -> result.method == method, results)
    switch_rate = mean(result.crossed for result in group)
    crossing_steps = [result.first_crossing for result in group if result.crossed]
    median_crossing = isempty(crossing_steps) ? -1.0 : median(crossing_steps)
    mean_loss = mean(result.final_loss for result in group)
    median_loss = median(result.final_loss for result in group)
    mean_calls = mean(result.solver_calls for result in group)
    summaries[method] = (
        switch_rate = switch_rate,
        median_crossing = median_crossing,
        mean_loss = mean_loss,
        median_loss = median_loss,
        mean_calls = mean_calls,
    )
    println(join((method, switch_rate, median_crossing, mean_loss, median_loss, mean_calls), '\t'))
end

# Go/no-go focuses on whether the proposal-faithful estimator repairs the dead
# local gradient. STE is a useful heuristic comparator, not a correctness oracle.
sage_summary = summaries[:sage]
hard_summary = summaries[:hard_local]
go = sage_summary.switch_rate >= 0.8 &&
     sage_summary.mean_loss < hard_summary.mean_loss &&
     hard_summary.switch_rate == 0.0

println()
println("Per-seed SAGE trajectories")
for result in filter(item -> item.method == :sage, results)
    println("seed=$(result.seed) crossed=$(result.crossed) first=$(result.first_crossing) final_z=$(result.final_z) final_loss=$(result.final_loss)")
end
println("DISCONTINUITY_LEARNING_GO_NO_GO=", go ? "PASS" : "FAIL")

