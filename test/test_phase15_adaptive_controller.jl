if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop

@testset "Phase 15 adaptive policy is frozen and validated" begin
    policy = Phase10JobShop.AdaptiveSAGEPolicy()
    @test Phase10JobShop.adaptive_sample_budget(0, policy) == 4
    @test Phase10JobShop.adaptive_sample_budget(1, policy) == 16
    @test Phase10JobShop.adaptive_sample_budget(2, policy) == 32
    @test Phase10JobShop.adaptive_sample_budget(4, policy) == 32
    @test_throws ArgumentError Phase10JobShop.adaptive_sample_budget(-1, policy)
    @test_throws ArgumentError Phase10JobShop.adaptive_sample_budget(5, policy)
    @test_throws ArgumentError Phase10JobShop.adaptive_sample_budget(
        0,
        Phase10JobShop.AdaptiveSAGEPolicy(probe_samples = 3),
    )
end

@testset "Phase 15 keeps probe and estimation accounting separate" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 4)
    problem = Phase10JobShop.prepare_problem(
        15; config = config, train_size = 3, iid_size = 2, ood_size = 2,
    )
    network = Phase10JobShop.pretrain_network(
        problem, 15; epochs = 1, batch_size = 3,
    )
    policy = Phase10JobShop.AdaptiveSAGEPolicy()
    estimate = Phase10JobShop.estimate_output_gradient(
        problem, network, 1, :sage_adaptive, 15_001;
        radius = 0.5, samples = 4, adaptive_policy = policy,
    )
    @test estimate.adaptive_stable_calls + estimate.adaptive_probed_calls == 1
    @test estimate.adaptive_probe_searches ==
        policy.probe_samples * estimate.adaptive_probed_calls
    @test estimate.adaptive_minimum_calls + estimate.adaptive_medium_calls +
        estimate.adaptive_maximum_calls == 1
    @test estimate.adaptive_estimation_samples in (
        policy.minimum_samples, policy.medium_samples, policy.maximum_samples,
    )
    @test estimate.global_searches >= estimate.adaptive_probe_searches
    @test isfinite(estimate.ess) && estimate.ess > 0
    @test all(isfinite, estimate.gradient)
    @test estimate.unresolved == 0
    @test estimate.logical_failures == 0
end
