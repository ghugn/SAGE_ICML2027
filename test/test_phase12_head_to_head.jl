if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop

@testset "Phase 12 switching exposure is shared, deterministic and exact" begin
    problem = Phase10JobShop.prepare_problem(
        4; train_size = 6, iid_size = 4, ood_size = 6,
    )
    pretrained = Phase10JobShop.pretrain_network(
        problem, 4; epochs = 1, batch_size = 3,
    )
    first = Phase10JobShop.switching_exposure_partition(
        pretrained, problem.ood, 4; radius = 0.5, samples = 4,
    )
    second = Phase10JobShop.switching_exposure_partition(
        pretrained, problem.ood, 4; radius = 0.5, samples = 4,
    )
    @test first == second
    @test isempty(intersect(first.lower, first.higher))
    @test sort(vcat(first.lower, first.higher)) == collect(1:6)
    @test length(first.lower) == length(first.higher) == 3
    @test all(score -> 0.0 <= score <= 1.0, first.scores)
    @test first.higher_rate >= first.lower_rate
    @test first.searches == 24
end

@testset "Phase 12 compares all five backwards from one initialization" begin
    problem = Phase10JobShop.prepare_problem(
        2; train_size = 5, iid_size = 3, ood_size = 3,
    )
    pretrained = Phase10JobShop.pretrain_network(
        problem, 2; epochs = 1, batch_size = 5,
    )
    frozen = deepcopy(pretrained)
    methods = (:hard_local, :ste, :blackbox, :sage_geometry, :sage_multiscale)
    results = Dict(method => Phase10JobShop.train_method(
        problem, pretrained, method, 2;
        max_steps = 1, samples = 4, radius = 0.5, evaluation_interval = 1,
    ) for method in methods)
    @test pretrained.W1 == frozen.W1
    @test pretrained.W2 == frozen.W2
    @test pretrained.W3 == frozen.W3
    @test all(result.steps == 1 for result in values(results))
    @test all(result.unresolved_samples == 0 for result in values(results))
    @test all(result.logical_failures == 0 for result in values(results))
    @test results[:blackbox].perturbed_global_searches == 4
    @test results[:sage_multiscale].base_proposal_samples +
        results[:sage_multiscale].local_proposal_samples +
        results[:sage_multiscale].focused_proposal_samples == 4
end
