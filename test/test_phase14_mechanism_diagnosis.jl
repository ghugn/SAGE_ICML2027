if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop

@testset "Phase 14 exposes reproducible output-gradient diagnostics" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 4)
    problem = Phase10JobShop.prepare_problem(
        14; config = config, train_size = 3, iid_size = 2, ood_size = 2,
    )
    network = Phase10JobShop.pretrain_network(
        problem, 14; epochs = 1, batch_size = 3,
    )

    first = Phase10JobShop.estimate_output_gradient(
        problem, network, 1, :blackbox, 1401; radius = 0.3, samples = 4,
    )
    repeat = Phase10JobShop.estimate_output_gradient(
        problem, network, 1, :blackbox, 1401; radius = 0.3, samples = 4,
    )
    @test first.gradient == repeat.gradient
    @test first.point == repeat.point
    @test first.global_searches == 4
    @test first.branch_solves > 0
    @test first.unresolved == 0
    @test first.logical_failures == 0
    @test all(isfinite, first.gradient)

    sage = Phase10JobShop.estimate_output_gradient(
        problem, network, 1, :sage_multiscale, 1402;
        radius = 0.3, samples = 4,
    )
    @test 0.0 < sage.ess <= 4.0
    @test sage.base_samples + sage.local_samples + sage.focused_samples == 4
    @test sage.unresolved == 0
    @test sage.logical_failures == 0
    @test all(isfinite, sage.gradient)

    local_gradient = Phase10JobShop.estimate_output_gradient(
        problem, network, 1, :hard_local, 1403;
        radius = 0.3, samples = 4,
    )
    @test local_gradient.global_searches == 0
    @test isnan(local_gradient.ess)
    @test_throws ArgumentError Phase10JobShop.estimate_output_gradient(
        problem, network, 1, :blackbox, 1404; radius = 0.3, samples = 3,
    )
end
