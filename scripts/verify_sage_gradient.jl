using DisjunctiveNet
using Random

function interval_union()
    model = DisjunctiveModel(1)
    set_bounds!(model, lower = [-2.0], upper = [2.0])
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, -1.0)],
        [LinearConstraint([1.0], :>=, 1.0)],
    )
    return model
end

function polynomial_ball_quantiles(count)
    cdf(u) = (15 / 16) * (u - 2u^3 / 3 + u^5 / 5 + 8 / 15)
    values = Float64[]
    for index in 1:count
        probability = (index - 0.5) / count
        lower = -1.0
        upper = 1.0
        for _ in 1:70
            midpoint = (lower + upper) / 2
            if cdf(midpoint) < probability
                lower = midpoint
            else
                upper = midpoint
            end
        end
        push!(values, (lower + upper) / 2)
    end
    return values
end

model = interval_union()
loss(y) = (y[1] - 1.5)^2
loss_gradient(y) = [2.0 * (y[1] - 1.5)]

# Stable fast path: no perturbed global search is allowed or needed.
stable_samples = reshape(range(-0.75, 0.75; length = 32) |> collect, 1, :)
stable = sage_gradient(
    model,
    [-1.5],
    0.2,
    loss,
    loss_gradient;
    samples = stable_samples,
    store_records = false,
)

# Proposal example with the C1 polynomial-ball density. The analytic gradient
# of the matching smoothed objective is -1.458.
sample_count = 1_000
quantiles = polynomial_ball_quantiles(sample_count)
switching = sage_gradient(
    model,
    [-0.4],
    0.5,
    loss,
    loss_gradient;
    samples = reshape(quantiles, 1, :),
    store_records = false,
)
analytic_gradient = -1.458
gradient_error = abs(only(switching.gradient) - analytic_gradient)

# Repeated Monte Carlo fidelity at a less rare switching location. This measures
# the complete estimator, including exact perturbed branch searches.
mc_center = [-0.1]
mc_radius = 0.5
mc_reference = -10.368
mc_repetitions = 20
mc_samples = 64
mc_center_result = lazy_branch_search(model, mc_center)
mc_certificate = switching_certificate(
    model,
    mc_center,
    mc_radius;
    center_result = mc_center_result,
)
mc_estimates = Float64[]
mc_global_searches = Ref(0)
for seed in 1:mc_repetitions
    estimate = sage_gradient(
        model,
        mc_center,
        mc_radius,
        loss,
        loss_gradient;
        n_samples = mc_samples,
        rng = MersenneTwister(20261113 + seed),
        center_result = mc_center_result,
        certificate = mc_certificate,
        store_records = false,
    )
    @assert estimate.status == :ok
    push!(mc_estimates, only(estimate.gradient))
    mc_global_searches[] += estimate.perturbed_global_searches
end
mc_mean = sum(mc_estimates) / mc_repetitions
mc_bias = mc_mean - mc_reference
mc_variance = sum((value - mc_mean)^2 for value in mc_estimates) / (mc_repetitions - 1)
mc_rmse = sqrt(sum((value - mc_reference)^2 for value in mc_estimates) / mc_repetitions)
mc_cosine_match = count(value -> sign(value) == sign(mc_reference), mc_estimates) / mc_repetitions

@assert stable.status == :ok
@assert stable.certificate.status == :stable
@assert stable.perturbed_global_searches == 0
@assert switching.status == :ok
@assert switching.certificate.status == :not_certified
@assert gradient_error < 0.03
@assert abs(mc_bias) < 2.0
@assert mc_cosine_match >= 0.9

println("DisjunctiveNet Phase 4 SAGE verification")
println("Julia version: ", VERSION)
println("Stable certificate status: ", stable.certificate.status)
println("Stable perturbed global searches: ", stable.perturbed_global_searches)
println("Stable reference branch solves: ", stable.reference_branch_solves)
println("Switching estimate: ", only(switching.gradient))
println("Analytic smoothed gradient: ", analytic_gradient)
println("Absolute gradient error: ", gradient_error)
println("Switching samples: ", switching.switching_samples, "/", sample_count)
println("Switching perturbed global searches: ", switching.perturbed_global_searches)
println("Switching competitor branch solves: ", switching.competitor_branch_solves)
println("MC repetitions/samples: ", mc_repetitions, "/", mc_samples)
println("MC analytic reference: ", mc_reference)
println("MC mean estimate: ", mc_mean)
println("MC bias: ", mc_bias)
println("MC variance: ", mc_variance)
println("MC RMSE: ", mc_rmse)
println("MC 1D cosine-sign match: ", mc_cosine_match)
println("MC total perturbed global searches: ", mc_global_searches[])
println("SAGE_GRADIENT_VERIFICATION=PASS")

