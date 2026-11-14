include(joinpath(@__DIR__, "..", "scripts", "phase9_cooling_common.jl"))
import .Phase9Cooling
using Random
using Statistics

@testset "Phase 9 cooling rules are input-dependent and exact-feasible" begin
    config = Phase9Cooling.CoolingConfig()
    hot = [35.0, 50.0, 0.4, 0.2, 0.0, 0.0, 0.0]
    @test Phase9Cooling.active_rule_names(hot, config) == [:hot]
    hot_model = Phase9Cooling.build_cooling_model(hot, config)
    @test length(hot_model.disjunctions) == 1
    @test hot_model.disjunctions[1].name == :hot

    combined = [35.0, 85.0, 0.9, 0.6, 1.0, 1.0, 1.0]
    rules = Phase9Cooling.active_rule_names(combined, config)
    @test rules == [
        :hot, :humid, :high_workload, :demand_response,
        :fan_maintenance, :chiller_maintenance,
    ]
    model = Phase9Cooling.build_cooling_model(combined, config)
    result = lazy_branch_search(model, [0.6, 0.6, 0.2]; store_prune_records = false)
    @test result.status == :optimal
    @test result.report !== nothing && result.report.feasible
    @test check_feasibility(model, result.y).feasible

    # Regression for a near-degenerate feasible branch that HiGHS reports as
    # OTHER_ERROR for the QP and the analytic active-set fallback resolves.
    cold_demand = [
        7.675884632521122, 74.03203139713537, 0.5835137122590583,
        0.4160657349718457, 1.0, 0.0, 0.0,
    ]
    retry_model = Phase9Cooling.build_cooling_model(cold_demand, config)
    retry_branch = project_branch(
        retry_model,
        [0.003814262845725086, -0.13353903851046123, -0.10538768483127478],
        [1, 1, 2],
    )
    @test retry_branch.accepted
    @test occursin("active-set fallback", something(retry_branch.message, ""))
    @test lazy_branch_search(
        retry_model,
        [0.003814262845725086, -0.13353903851046123, -0.10538768483127478];
        store_prune_records = false,
    ).status == :optimal
end

@testset "Phase 9 reconstructed data are deterministic and train-normalized" begin
    first = Phase9Cooling.prepare_problem(7; train_size = 16, iid_size = 8, ood_size = 8)
    second = Phase9Cooling.prepare_problem(7; train_size = 16, iid_size = 8, ood_size = 8)
    @test first.train.raw_inputs == second.train.raw_inputs
    @test first.train.targets == second.train.targets
    @test first.iid.raw_inputs == second.iid.raw_inputs
    @test first.ood.raw_inputs == second.ood.raw_inputs
    @test first.train.oracle_choices == second.train.oracle_choices
    @test vec(mean(first.train.inputs; dims = 2)) ≈ zeros(7) atol = 1e-12
    @test all(isfinite, first.train.inputs)
    for split in (first.train, first.iid, first.ood), index in eachindex(split.models)
        @test check_feasibility(split.models[index], split.oracle_outputs[:, index]).feasible
    end
end

@testset "Phase 9 matched initialization and all backward paths" begin
    problem = Phase9Cooling.prepare_problem(3; train_size = 12, iid_size = 6, ood_size = 6)
    pretrained = Phase9Cooling.pretrain_network(problem, 3; epochs = 1, batch_size = 4)
    duplicate = deepcopy(pretrained)
    @test pretrained.W1 == duplicate.W1
    @test pretrained.W2 == duplicate.W2
    @test pretrained.W3 == duplicate.W3

    results = Dict(method => Phase9Cooling.train_method(
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
        Phase9Cooling.evaluate_network(result.network, problem.iid).csat == 1.0
        for result in values(results)
    )
end
