using Random
using LinearAlgebra

@testset "Geometry proposal preserves the compact-q target" begin
    dimension = 2
    beta = 2.0
    normals = [1.0 0.0; 0.0 1.0]
    count = 20_000
    batch = sample_geometry_smoothing(
        MersenneTwister(2026111401),
        dimension,
        count,
        normals;
        beta = beta,
        mixture_weight = 0.5,
        focus_strength = 0.9,
    )

    @test size(batch.samples) == (dimension, count)
    @test all(sum(abs2, batch.samples[:, index]) < 1.0 for index in 1:count)
    @test all(isfinite, batch.importance_weights)
    @test all(batch.importance_weights .> 0.0)
    @test all(isapprox.(
        batch.importance_weights .* batch.proposal_ratios,
        1.0;
        atol = 1e-12,
    ))
    @test :base in batch.components
    @test :focused in batch.components

    weighted_mass = sum(batch.importance_weights) / count
    weighted_mean = batch.samples * batch.importance_weights ./ count
    expected_coordinate_second_moment = 1.0 / (dimension + 2beta + 2.0)
    weighted_second_moments = [
        sum(batch.importance_weights .* abs2.(batch.samples[row, :])) / count
        for row in 1:dimension
    ]
    @test isapprox(weighted_mass, 1.0; atol = 0.02)
    @test norm(weighted_mean) < 0.02
    @test all(isapprox(
        moment,
        expected_coordinate_second_moment;
        atol = 0.015,
    ) for moment in weighted_second_moments)

    q_rng = MersenneTwister(2026111402)
    ratio_mean = sum(
        geometry_proposal_ratio(
            sample_ball_smoothing(q_rng, dimension; beta = beta),
            normals;
            beta = beta,
            mixture_weight = 0.5,
            focus_strength = 0.9,
        ) for _ in 1:count
    ) / count
    @test isapprox(ratio_mean, 1.0; atol = 0.02)
end

@testset "Certified multiscale proposal preserves the same compact-q target" begin
    dimension = 3
    beta = 2.0
    normals = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    local_scale = 0.35
    count = 30_000
    batch = sample_certified_geometry_smoothing(
        MersenneTwister(2026111411),
        dimension,
        count,
        normals;
        beta = beta,
        base_weight = 0.2,
        local_weight = 0.4,
        geometry_weight = 0.4,
        local_scale = local_scale,
        focus_strength = 0.9,
    )

    @test size(batch.samples) == (dimension, count)
    @test all(batch.importance_weights .* batch.proposal_ratios .≈ 1.0)
    @test Set(batch.components) == Set((:base, :local, :focused))
    @test all(
        norm(batch.samples[:, index]) < local_scale
        for index in eachindex(batch.components)
        if batch.components[index] == :local
    )
    weighted_mass = sum(batch.importance_weights) / count
    weighted_mean = batch.samples * batch.importance_weights ./ count
    expected_second_moment = 1.0 / (dimension + 2beta + 2.0)
    weighted_second_moments = [
        sum(batch.importance_weights .* abs2.(batch.samples[row, :])) / count
        for row in 1:dimension
    ]
    @test isapprox(weighted_mass, 1.0; atol = 0.025)
    @test norm(weighted_mean) < 0.025
    @test all(isapprox(moment, expected_second_moment; atol = 0.018)
              for moment in weighted_second_moments)

    @test scaled_ball_proposal_ratio([0.2, 0.0], 1.0) ≈ 1.0
    @test scaled_ball_proposal_ratio([0.5, 0.0], 0.4) == 0.0
    q_rng = MersenneTwister(2026111412)
    ratio_mean = sum(
        certified_geometry_proposal_ratio(
            sample_ball_smoothing(q_rng, dimension; beta = beta),
            normals;
            beta = beta,
            base_weight = 0.2,
            local_weight = 0.4,
            geometry_weight = 0.4,
            local_scale = local_scale,
            focus_strength = 0.9,
        ) for _ in 1:count
    ) / count
    @test isapprox(ratio_mean, 1.0; atol = 0.03)
end

@testset "Optimizer geometry follows proposal equation 21" begin
    model = DisjunctiveModel(1)
    set_bounds!(model, lower = [-2.0], upper = [2.0])
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, -1.0)],
        [LinearConstraint([1.0], :>=, 1.0)],
    )
    center = [-0.1]
    radius = 0.5
    center_result = lazy_branch_search(model, center; store_prune_records = false)
    certificate = switching_certificate(
        model,
        center,
        radius;
        center_result = center_result,
    )
    geometry = switching_geometry(
        model,
        center,
        certificate.winner,
        certificate.winner_point,
        competitor_set(certificate),
    )

    @test geometry.status == :ok
    @test geometry.branch_solves == 1
    @test geometry.accepted_branches == 1
    @test size(geometry.normals) == (1, 1)
    @test isapprox(geometry.normals[1, 1], 1.0; atol = 1e-8)
    @test geometry.competitor_choices == [[2]]
end

@testset "Geometry-aware SAGE is exact-feasible and importance weighted" begin
    model = DisjunctiveModel(1)
    set_bounds!(model, lower = [-2.0], upper = [2.0])
    add_disjunction!(
        model,
        [LinearConstraint([1.0], :<=, -1.0)],
        [LinearConstraint([1.0], :>=, 1.0)],
    )
    loss(y) = (y[1] - 1.5)^2
    loss_gradient(y) = [2.0 * (y[1] - 1.5)]
    config = SAGEConfig(
        radius = 0.5,
        n_samples = 64,
        sampling = :geometry,
        store_records = true,
    )
    estimate = sage_gradient(
        model,
        [-0.1],
        loss,
        loss_gradient,
        config;
        rng = MersenneTwister(2026111403),
    )

    @test estimate.status == :ok
    @test estimate.sampling == :geometry
    @test estimate.geometry_status == :ok
    @test estimate.geometry_normals == 1
    @test estimate.geometry_branch_solves == 1
    @test 0.0 < estimate.importance_effective_sample_size <= estimate.samples
    @test any(!isapprox(record.importance_weight, 1.0; atol = 1e-12)
              for record in estimate.records)
    @test all(record.status == :ok for record in estimate.records)
    @test all(is_logically_feasible(model, record.global_y)
              for record in estimate.records)

    reconstructed = sum(
        record.importance_weight .* (record.local_gradient .+ record.correction)
        for record in estimate.records
    ) ./ estimate.samples
    @test isapprox(estimate.gradient, reconstructed; atol = 1e-12)

    unpruned = sage_gradient(
        model,
        [-0.1],
        0.5,
        loss,
        loss_gradient;
        samples = reshape([-0.8, -0.4, 0.1, 0.4, 0.8], 1, :),
        use_ball_certificate = false,
        use_sample_certificate = false,
        store_records = false,
    )
    @test unpruned.status == :ok
    @test unpruned.perturbed_global_searches == 5
    @test unpruned.sample_stability_certificates == 0

    stable = sage_gradient(
        model,
        [-1.5],
        0.2,
        loss,
        loss_gradient;
        n_samples = 8,
        sampling = :geometry,
        rng = MersenneTwister(2026111404),
        store_records = false,
    )
    @test stable.status == :ok
    @test stable.sampling == :geometry_fallback
    @test stable.geometry_status == :stable_fallback
    @test stable.geometry_branch_solves == 0
    @test stable.perturbed_global_searches == 0
end

@testset "Geometry proposal validates unsafe configurations" begin
    @test_throws ArgumentError geometry_proposal_ratio([0.0], zeros(1, 1))
    @test_throws ArgumentError geometry_proposal_ratio(
        [0.0], ones(1, 1); mixture_weight = 1.0,
    )
    @test_throws ArgumentError sample_geometry_smoothing(
        MersenneTwister(1), 1, 4, ones(1, 1); focus_strength = 1.1,
    )
    @test_throws ArgumentError scaled_ball_proposal_ratio([0.0], 0.0)
    @test_throws ArgumentError certified_geometry_proposal_ratio(
        [0.0], ones(1, 1); base_weight = 0.0,
    )
    @test_throws ArgumentError certified_geometry_proposal_ratio(
        [0.0], ones(1, 1); base_weight = 0.2, local_weight = 0.2,
        geometry_weight = 0.2,
    )
end
