include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
import .Phase10JobShop
using Statistics

@testset "Phase 10 job-shop uses native machine-order disjunctions" begin
    config = Phase10JobShop.JobShopConfig()
    durations = fill(1.0 / 9.0, 9)
    machines = [1, 2, 3, 2, 3, 1, 3, 1, 2]
    model = Phase10JobShop.build_jobshop_model(durations, machines, config)
    @test length(model.disjunctions) == 9
    @test num_branches(model) == 512
    @test length(model.constraints) == 15
    @test all(rule -> startswith(string(rule.name), "machine_"), model.disjunctions)
end

@testset "Phase 10 high-dimensional QP fallback resolves a feasible branch" begin
    problem = Phase10JobShop.prepare_problem(
        1; train_size = 16, iid_size = 12, ood_size = 12,
    )
    # Seed 1, frozen pilot step 5 selects training instance 7. The geometry
    # sampler exposed the following feasible branch/point pair there.
    instance = problem.train.instances[7]
    point = [
        -0.336450745403871, -0.06937567324466365, 0.07025106050511994,
        0.4615504378282294, 0.7519172367609113, 0.5938927094082025,
        1.4506605972589672, 1.0196701139382236, 0.22279462429326438,
    ]
    choices = [1, 1, 1, 2, 2, 1, 1, 1, 2]
    branch = project_branch(instance.model, point, choices)
    @test branch.accepted
    @test branch.report !== nothing && branch.report.feasible
    @test occursin("primal active-set fallback", something(branch.message, ""))
end

@testset "Phase 10 generated schedules are deterministic and exact-feasible" begin
    first = Phase10JobShop.prepare_problem(
        7; train_size = 3, iid_size = 2, ood_size = 2,
    )
    second = Phase10JobShop.prepare_problem(
        7; train_size = 3, iid_size = 2, ood_size = 2,
    )
    @test first.train.raw_inputs == second.train.raw_inputs
    @test first.train.targets == second.train.targets
    @test first.train.instances[1].oracle_choices ==
        second.train.instances[1].oracle_choices
    @test vec(mean(first.train.inputs; dims = 2)) ≈ zeros(36) atol = 1e-10
    for split in (first.train, first.iid, first.ood), instance in split.instances
        @test check_feasibility(instance.model, instance.oracle_starts).feasible
        @test check_feasibility(instance.model, instance.heuristic_starts).feasible
        @test instance.oracle_makespan <= instance.heuristic_makespan + 1e-8
        @test length(instance.oracle_choices) == 9
    end
end

@testset "Phase 10 matched initialization and backward paths" begin
    problem = Phase10JobShop.prepare_problem(
        3; train_size = 3, iid_size = 2, ood_size = 2,
    )
    pretrained = Phase10JobShop.pretrain_network(
        problem, 3; epochs = 1, batch_size = 3,
    )
    duplicate = deepcopy(pretrained)
    @test pretrained.W1 == duplicate.W1
    @test pretrained.W2 == duplicate.W2
    @test pretrained.W3 == duplicate.W3

    results = Dict(method => Phase10JobShop.train_method(
        problem,
        pretrained,
        method,
        3;
        max_steps = 1,
        samples = 2,
        evaluation_interval = 1,
    ) for method in (:hard_local, :ste, :blackbox, :sage_cert, :sage_geometry))
    @test results[:blackbox].perturbed_global_searches == 2
    @test 0 <= results[:sage_cert].perturbed_global_searches <= 2
    @test 0 <= results[:sage_geometry].perturbed_global_searches <= 2
    @test all(result.unresolved_samples == 0 for result in values(results))
    @test all(result.logical_failures == 0 for result in values(results))
    @test all(
        Phase10JobShop.evaluate_network(
            result.network, problem.iid; config = problem.config,
        ).csat == 1.0
        for result in values(results)
    )
end
