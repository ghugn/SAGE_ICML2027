if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop

@testset "Phase 13 generalizes the exact Job-Shop family to 3x4" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 4)
    problem = Phase10JobShop.prepare_problem(
        6; config = config, train_size = 3, iid_size = 2, ood_size = 2,
    )
    @test size(problem.train.inputs) == (60, 3)
    @test size(problem.train.targets) == (12, 3)
    @test all(length(instance.model.disjunctions) == 12
        for split in (problem.train, problem.iid, problem.ood)
        for instance in split.instances)
    @test all(num_branches(instance.model) == 4096
        for split in (problem.train, problem.iid, problem.ood)
        for instance in split.instances)
    @test all(check_feasibility(instance.model, instance.oracle_starts).feasible
        for split in (problem.train, problem.iid, problem.ood)
        for instance in split.instances)

    network = Phase10JobShop.pretrain_network(
        problem, 6; epochs = 1, batch_size = 3,
    )
    @test size(network.W1) == (config.hidden, 60)
    @test size(network.W3) == (12, config.hidden)
    evaluation = Phase10JobShop.evaluate_network(
        network, problem.ood; config = config,
    )
    @test evaluation.csat == 1.0
    @test evaluation.unresolved == 0
end

@testset "Phase 13 gradient audit is finite and proposal-faithful" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 4)
    problem = Phase10JobShop.prepare_problem(
        2; config = config, train_size = 3, iid_size = 2, ood_size = 2,
    )
    network = Phase10JobShop.pretrain_network(
        problem, 2; epochs = 1, batch_size = 3,
    )
    audit = Phase10JobShop.output_gradient_audit(
        problem, network, 1, 2;
        radius = 0.5, reference_samples = 4, sage_samples = 4,
    )
    @test audit.reference_norm > 0
    @test audit.reference.global_searches == 4
    @test all(isfinite, audit.reference.gradient)
    for item in (audit.hard_local, audit.ste, audit.sage_multiscale)
        @test isfinite(item.cosine)
        @test isfinite(item.relative_error)
        @test item.unresolved == 0
        @test item.logical_failures == 0
    end
    @test 0.0 < audit.sage_multiscale.ess <= 4.0
end

@testset "Phase 13 protects the controlled combinatorial envelope" begin
    @test_throws ArgumentError Phase10JobShop.prepare_problem(
        1;
        config = Phase10JobShop.JobShopConfig(n_jobs = 4, n_machines = 4),
        train_size = 2, iid_size = 2, ood_size = 2,
    )
end
