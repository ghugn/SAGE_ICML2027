if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop

@testset "Phase 16 checkpoints come from one uninterrupted run" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 3)
    problem = Phase10JobShop.prepare_problem(
        16; config = config, train_size = 3, iid_size = 2, ood_size = 2,
    )
    network = Phase10JobShop.pretrain_network(
        problem, 16; epochs = 1, batch_size = 3,
    )
    reported_steps = Int[]
    result = Phase10JobShop.train_method(
        problem, network, :ste, 16;
        epochs = 1, max_steps = 2, samples = 2,
        evaluation_interval = 10, checkpoint_steps = [1, 2],
        progress_callback = item -> push!(reported_steps, item.step),
    )
    @test sort!(collect(keys(result.checkpoint_networks))) == [1, 2]
    @test [item.step for item in result.trajectory] == [0, 1, 2]
    @test result.trajectory[1].elapsed_seconds == 0.0
    @test result.trajectory[2].elapsed_seconds >= 0.0
    @test result.trajectory[3].elapsed_seconds >=
        result.trajectory[2].elapsed_seconds
    @test all(item.perturbed_global_searches == 0 for item in result.trajectory)
    @test all(item.complete_branch_solves >= 0 for item in result.trajectory)
    @test reported_steps == [0, 1, 2]
    final_evaluation = Phase10JobShop.evaluate_network(
        result.checkpoint_networks[2], problem.ood; config = config,
    )
    @test final_evaluation.csat == 1.0
    @test final_evaluation.unresolved == 0

    @test_throws ArgumentError Phase10JobShop.train_method(
        problem, network, :ste, 16;
        epochs = 1, max_steps = 2, samples = 2,
        evaluation_interval = 10, checkpoint_steps = [3],
    )
end
