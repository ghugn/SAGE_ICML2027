if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop

@testset "Phase 11 multiscale path is certified, weighted and exact-feasible" begin
    problem = Phase10JobShop.prepare_problem(
        1; train_size = 8, iid_size = 4, ood_size = 4,
    )
    pretrained = Phase10JobShop.pretrain_network(
        problem, 1; epochs = 2, batch_size = 4,
    )
    result = Phase10JobShop.train_method(
        problem,
        pretrained,
        :sage_multiscale,
        1;
        max_steps = 1,
        samples = 4,
        radius = 0.5,
        evaluation_interval = 1,
    )
    @test result.unresolved_samples == 0
    @test result.logical_failures == 0
    @test result.perturbed_global_searches <= 4
    @test result.positive_local_radius_calls == 1
    @test result.mean_local_radius > 0.0
    @test result.local_proposal_samples > 0
    @test result.base_proposal_samples + result.local_proposal_samples +
        result.focused_proposal_samples == 4
    @test 0.0 < result.mean_effective_sample_size <= 4.0
    iid = Phase10JobShop.evaluate_network(
        result.network, problem.iid; config = problem.config,
    )
    ood = Phase10JobShop.evaluate_network(
        result.network, problem.ood; config = problem.config,
    )
    @test iid.csat == 1.0
    @test ood.csat == 1.0
    @test iid.unresolved == 0
    @test ood.unresolved == 0
end
