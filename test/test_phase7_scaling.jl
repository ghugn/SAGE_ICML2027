include(joinpath(@__DIR__, "..", "scripts", "phase7_scaling_common.jl"))
using .Phase7Scaling
using Random
using LinearAlgebra

@testset "Phase 7 deterministic scaling family" begin
    for dimension in (2, 4, 8), branches in (4, 8, 16)
        geometry = scaling_box_union(dimension, branches)
        repeated = scaling_box_union(dimension, branches)
        @test size(geometry.centers) == (dimension, branches)
        @test geometry.centers == repeated.centers
        @test minimum_center_distance(geometry, Inf) > 2geometry.halfwidth
        @test all(isfinite, geometry.centers)
    end

    @test_throws ArgumentError scaling_box_union(0, 4)
    @test_throws ArgumentError scaling_box_union(2, 1)
end

@testset "Phase 7 analytic oracle matches certified backend" begin
    geometry = scaling_box_union(2, 4)
    model = disjunctive_model(geometry)
    for point in (
        copy(geometry.centers[:, 1]),
        copy(geometry.centers[:, 4]),
        0.52 .* geometry.centers[:, 1] .+ 0.48 .* geometry.centers[:, 2],
    )
        analytic = analytic_project(geometry, point)
        backend = lazy_branch_search(model, point; store_prune_records = false)
        @test backend.status == :optimal
        @test only(backend.choices) == analytic.branch
        @test isapprox(backend.y, analytic.y; atol = 5e-7)
        @test backend.report !== nothing && backend.report.feasible
    end
end

@testset "Phase 7 unpruned and certified SAGE preserve equation 17-18" begin
    geometry = scaling_box_union(2, 4)
    model = disjunctive_model(geometry)
    target = copy(geometry.centers[:, 4])
    z = 0.52 .* geometry.centers[:, 1] .+ 0.48 .* geometry.centers[:, 2]
    radius = 0.40 * norm(geometry.centers[:, 1] .- geometry.centers[:, 2])
    samples = sample_matrix(MersenneTwister(20271113), 2, 12)
    center = lazy_branch_search(model, z; store_prune_records = false)
    analytic = analytic_sage_gradient(geometry, z, radius, target, samples)

    unpruned = backend_sage_unpruned_gradient(
        model,
        geometry,
        center,
        z,
        radius,
        target,
        samples,
    )
    certified = sage_gradient(
        model,
        z,
        radius,
        y -> task_loss(y, target),
        y -> task_loss_gradient(y, target);
        samples = samples,
        center_result = center,
        store_records = true,
    )

    @test certified.status == :ok
    @test isapprox(unpruned.gradient, analytic.gradient; atol = 1e-6)
    @test isapprox(certified.gradient, analytic.gradient; atol = 1e-6)
    @test unpruned.unsafe_pruning == 0
    @test unpruned.feasible
    @test all(record.status == :ok for record in certified.records)
    @test all(check_feasibility(model, record.global_y).feasible for record in certified.records)
    @test all(begin
        exact = analytic_project(geometry, record.w)
        only(record.global_choices) == exact.branch &&
            isapprox(record.global_y, exact.y; atol = 5e-7)
    end for record in certified.records)

    stable_z = copy(geometry.centers[:, 1])
    stable_radius = 0.10 * minimum_center_distance(geometry)
    stable_center = lazy_branch_search(model, stable_z; store_prune_records = false)
    stable = sage_gradient(
        model,
        stable_z,
        stable_radius,
        y -> task_loss(y, target),
        y -> task_loss_gradient(y, target);
        samples = samples,
        center_result = stable_center,
        store_records = true,
    )
    @test stable.status == :ok
    @test certifies_stability(stable.certificate)
    @test stable.perturbed_global_searches == 0
    @test stable.competitor_branch_solves == 0
end

@testset "Phase 7 exact lazy fallback avoids competitor enumeration" begin
    geometry = scaling_box_union(4, 16)
    model = disjunctive_model(geometry)
    target = copy(geometry.centers[:, 16])
    z = 0.52 .* geometry.centers[:, 1] .+ 0.48 .* geometry.centers[:, 2]
    radius = 0.40 * norm(geometry.centers[:, 1] .- geometry.centers[:, 2])
    samples = sample_matrix(MersenneTwister(20271114), 4, 6; antithetic = true)
    center = lazy_branch_search(model, z; store_prune_records = false)
    analytic = analytic_sage_gradient(geometry, z, radius, target, samples)
    estimate = sage_gradient(
        model,
        z,
        radius,
        y -> task_loss(y, target),
        y -> task_loss_gradient(y, target);
        samples = samples,
        center_result = center,
        global_strategy = :lazy,
        store_records = true,
    )
    @test estimate.status == :ok
    @test estimate.lazy_global_searches == size(samples, 2)
    @test estimate.restricted_global_searches == 0
    @test isapprox(estimate.gradient, analytic.gradient; atol = 1e-6)
    @test all(check_feasibility(model, record.global_y).feasible for record in estimate.records)
end
