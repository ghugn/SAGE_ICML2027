if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end

@testset "Interrupted training resumes exactly" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 3)
    problem = Phase10JobShop.prepare_problem(
        160; config = config, train_size = 4, iid_size = 2, ood_size = 2,
    )
    network = Phase10JobShop.pretrain_network(
        problem, 160; epochs = 1, batch_size = 2,
    )
    uninterrupted = Phase10JobShop.train_method(
        problem, network, :ste, 160;
        epochs = 2, max_steps = 4, samples = 2,
        evaluation_interval = 2, checkpoint_steps = [2, 4],
    )
    snapshots = Any[]
    Phase10JobShop.train_method(
        problem, network, :ste, 160;
        epochs = 1, max_steps = 2, samples = 2,
        evaluation_interval = 2, checkpoint_steps = [2],
        state_checkpoint_interval = 2,
        state_checkpoint_callback = snapshot -> push!(snapshots, snapshot),
    )
    @test length(snapshots) == 1
    resumed = Phase10JobShop.train_method(
        problem, network, :ste, 160;
        epochs = 2, max_steps = 4, samples = 2,
        evaluation_interval = 2, checkpoint_steps = [2, 4],
        resume_state = only(snapshots),
    )
    for name in (:W1, :b1, :W2, :b2, :W3, :b3)
        @test getproperty(resumed.network, name) ==
            getproperty(uninterrupted.network, name)
    end
    @test resumed.steps == uninterrupted.steps == 4
    @test resumed.perturbed_global_searches ==
        uninterrupted.perturbed_global_searches
    @test resumed.complete_branch_solves == uninterrupted.complete_branch_solves
    @test sort!(collect(keys(resumed.checkpoint_networks))) == [2, 4]
    @test [item.step for item in resumed.trajectory] == [0, 2, 4]
    @test resumed.training_seconds >= only(snapshots).elapsed_seconds

    @test_throws ArgumentError Phase10JobShop.train_method(
        problem, network, :ste, 160;
        epochs = 1, max_steps = 2, samples = 2,
        state_checkpoint_interval = 1,
    )
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
