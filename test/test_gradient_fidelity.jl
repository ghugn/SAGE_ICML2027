using Random
using LinearAlgebra

@testset "SAGE smoothing density and score" begin
    score = ball_smoothing_score([0.3, 0.4]; beta = 2.0)
    @test isapprox(score, [1.6, 2.1333333333333333]; atol = 1e-12)
    @test ball_smoothing_score([0.0, 0.0]) == [0.0, 0.0]

    rng = MersenneTwister(20261113)
    samples = [sample_ball_smoothing(rng, 3; beta = 2.0) for _ in 1:10_000]
    @test all(norm(sample) < 1.0 for sample in samples)
    empirical_radius_squared = sum(sum(abs2, sample) for sample in samples) / length(samples)
    expected_radius_squared = (3 / 2) / (3 / 2 + 3)
    @test isapprox(empirical_radius_squared, expected_radius_squared; atol = 0.01)

    @test_throws ArgumentError sample_ball_smoothing(rng, 0)
    @test_throws ArgumentError sample_ball_smoothing(rng, 2; beta = 1.5)
    @test_throws ArgumentError ball_smoothing_score([1.0])

    antithetic = sample_antithetic_ball_smoothing(MersenneTwister(7), 3, 7)
    @test size(antithetic) == (3, 7)
    @test antithetic[:, 2] == -antithetic[:, 1]
    @test antithetic[:, 4] == -antithetic[:, 3]
    @test antithetic[:, 6] == -antithetic[:, 5]
    @test all(norm(antithetic[:, column]) < 1.0 for column in axes(antithetic, 2))
    @test_throws ArgumentError sample_antithetic_ball_smoothing(rng, 2, 0)
    @test_throws ArgumentError sample_antithetic_ball_smoothing(rng, 0, 2)
    @test_throws ArgumentError sample_antithetic_ball_smoothing(rng, 2, 2; beta = 1.5)
end

@testset "Fixed-branch implicit gradient matches finite differences" begin
    equality_model = DisjunctiveModel(2)
    add_linear_constraint!(equality_model, [1.0, 1.0], :(==), 1.0)
    w = [0.8, 0.6]
    target = [0.1, 0.2]
    loss(y) = 0.5 * sum(abs2, y .- target)
    loss_gradient(y) = y .- target

    gradient_result = fixed_branch_loss_gradient(
        equality_model,
        w,
        Int[],
        loss_gradient,
    )
    @test gradient_result.status == :ok
    @test gradient_result.active_constraints == 1
    @test gradient_result.active_rank == 1
    @test isapprox(
        gradient_result.jacobian,
        [0.5 -0.5; -0.5 0.5];
        atol = 1e-7,
    )

    epsilon = 1e-5
    finite_difference = zeros(2)
    for coordinate in 1:2
        direction = zeros(2)
        direction[coordinate] = epsilon
        plus = project_branch(equality_model, w .+ direction, Int[])
        minus = project_branch(equality_model, w .- direction, Int[])
        finite_difference[coordinate] = (loss(plus.y) - loss(minus.y)) / (2epsilon)
    end
    @test isapprox(gradient_result.gradient, finite_difference; atol = 2e-5)

    interval_model = DisjunctiveModel(1)
    add_disjunction!(
        interval_model,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)],
    )
    interior = fixed_branch_loss_gradient(interval_model, [1.0], [1], y -> [2y[1]])
    saturated = fixed_branch_loss_gradient(interval_model, [3.0], [1], y -> [2y[1]])
    @test isapprox(interior.gradient, [2.0]; atol = 1e-6)
    @test isapprox(interior.jacobian, ones(1, 1); atol = 1e-8)
    @test isapprox(saturated.gradient, [0.0]; atol = 1e-8)
    @test isapprox(saturated.jacobian, zeros(1, 1); atol = 1e-8)
end

@testset "Certified competitor set gives exact perturbed winner" begin
    model = DisjunctiveModel(2)
    set_bounds!(model, lower = zeros(2), upper = ones(2))
    for coordinate in 1:2
        direction = zeros(2)
        direction[coordinate] = 1.0
        add_disjunction!(
            model,
            [LinearConstraint(direction, :<=, 0.2)],
            [LinearConstraint(direction, :>=, 0.8)],
        )
    end

    center = [0.9, 0.5]
    radius = 0.05
    certificate = switching_certificate(model, center, radius)
    @test certificate.status == :not_certified
    @test certificate.total_competitor_branches == 3
    @test certificate.stable_pruned_branches == 2
    @test certificate.competitive_branches == 1

    for dy in range(-radius, radius; length = 21)
        w = center .+ [0.0, dy]
        relevant = project_relevant_branches(
            model,
            w,
            certificate.winner,
            competitor_set(certificate),
        )
        oracle = exhaustive_project(model, w)
        @test relevant.status == :optimal
        @test relevant.choices == oracle.choices
        @test isapprox(relevant.objective, oracle.objective; atol = 1e-8)
        @test relevant.solved_branches == 2
    end
end

@testset "SAGE stable fast path follows equation 17" begin
    model = DisjunctiveModel(1)
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)],
    )
    loss(y) = y[1]^2
    loss_gradient(y) = [2y[1]]
    samples = reshape([-0.5, 0.5], 1, 2)
    estimate = sage_gradient(
        model,
        [1.0],
        0.2,
        loss,
        loss_gradient;
        samples = samples,
    )

    @test estimate.status == :ok
    @test is_certified(estimate)
    @test estimate.certificate.status == :stable
    @test isapprox(estimate.gradient, [2.0]; atol = 1e-6)
    @test estimate.reference_branch_solves == 2
    @test estimate.perturbed_global_searches == 0
    @test estimate.competitor_branch_solves == 0
    @test estimate.switching_samples == 0
    @test all(record.residual == 0.0 for record in estimate.records)
    @test all(record.correction == [0.0] for record in estimate.records)
end

@testset "SAGE residual correction follows proposal equation 18" begin
    model = DisjunctiveModel(1)
    set_bounds!(model, lower = [-2.0], upper = [2.0])
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, -1.0)],
        [LinearConstraint([1.0], :>=, 1.0)],
    )
    loss(y) = (y[1] - 1.5)^2
    loss_gradient(y) = [2.0 * (y[1] - 1.5)]

    estimate = sage_gradient(
        model,
        [-0.4],
        0.5,
        loss,
        loss_gradient;
        samples = reshape([0.9], 1, 1),
    )
    record = only(estimate.records)
    expected_score = 4.0 * 0.9 / (1.0 - 0.9^2)
    expected_correction = -6.0 * expected_score / 0.5

    @test estimate.status == :ok
    @test estimate.certificate.status == :not_certified
    @test estimate.winner == [1]
    @test estimate.perturbed_global_searches == 1
    @test estimate.competitor_branch_solves == 1
    @test estimate.switching_samples == 1
    @test record.global_choices == [2]
    @test record.local_y ≈ [-1.0]
    @test record.global_y ≈ [1.0]
    @test record.local_loss ≈ 6.25
    @test record.global_loss ≈ 0.25
    @test record.residual ≈ -6.0
    @test record.local_gradient ≈ [0.0]
    @test record.correction ≈ [expected_correction]
    @test estimate.gradient ≈ [expected_correction]

    @test_throws ArgumentError sage_gradient(
        model,
        [-0.4],
        0.0,
        loss,
        loss_gradient,
    )
    @test_throws ArgumentError sage_gradient(
        model,
        [-0.4],
        0.5,
        loss,
        loss_gradient;
        samples = reshape([1.0], 1, 1),
    )
    stale_certificate = switching_certificate(model, [-0.4], 0.4)
    @test_throws ArgumentError sage_gradient(
        model,
        [-0.4],
        0.5,
        loss,
        loss_gradient;
        certificate = stale_certificate,
        samples = reshape([0.9], 1, 1),
    )
end

@testset "SAGE matches the analytic two-interval smoothed gradient" begin
    model = DisjunctiveModel(1)
    set_bounds!(model, lower = [-2.0], upper = [2.0])
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, -1.0)],
        [LinearConstraint([1.0], :>=, 1.0)],
    )
    loss(y) = (y[1] - 1.5)^2
    loss_gradient(y) = [2.0 * (y[1] - 1.5)]

    # Deterministic midpoint quantiles of q(u) = 15/16*(1-u^2)^2 on [-1,1].
    cdf(u) = (15 / 16) * (u - 2u^3 / 3 + u^5 / 5 + 8 / 15)
    function quantile(probability)
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
        return (lower + upper) / 2
    end
    sample_count = 1_000
    quantiles = [quantile((index - 0.5) / sample_count) for index in 1:sample_count]
    estimate = sage_gradient(
        model,
        [-0.4],
        0.5,
        loss,
        loss_gradient;
        samples = reshape(quantiles, 1, :),
        store_records = false,
    )

    # Switching occurs at u > 0.8. Integrating equation (17) gives
    # -24*(15/16)/0.5 * integral_0.8^1 u*(1-u^2) du = -1.458.
    analytic_gradient = -1.458
    @test estimate.status == :ok
    @test isapprox(only(estimate.gradient), analytic_gradient; atol = 0.03)
    @test estimate.switching_samples == 9
    @test estimate.perturbed_global_searches == sample_count
    @test estimate.competitor_branch_solves == sample_count
end

