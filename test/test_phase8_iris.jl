include(joinpath(@__DIR__, "..", "scripts", "phase8_iris_common.jl"))
using .Phase8Iris
using LinearAlgebra
using Random
using Statistics

@testset "Phase 8 Iris data split and train-only geometry" begin
    path = joinpath(@__DIR__, "..", "data", "phase8", "iris.data")
    dataset = load_iris(path)
    @test size(dataset.inputs) == (2, 150)
    @test size(dataset.targets) == (2, 150)
    @test count(==(1), dataset.labels) == 50
    @test count(==(2), dataset.labels) == 50
    @test count(==(3), dataset.labels) == 50

    problem = prepare_problem(dataset, 1)
    @test length(problem.train) == 90
    @test length(problem.validation) == 30
    @test length(problem.test) == 30
    @test isempty(intersect(problem.train, problem.validation))
    @test isempty(intersect(problem.train, problem.test))
    @test isempty(intersect(problem.validation, problem.test))
    @test all(count(index -> problem.labels[index] == label, problem.train) == 30 for label in 1:3)
    @test all(count(index -> problem.labels[index] == label, problem.validation) == 10 for label in 1:3)
    @test all(count(index -> problem.labels[index] == label, problem.test) == 10 for label in 1:3)
    @test vec(mean(problem.inputs[:, problem.train]; dims = 2)) ≈ zeros(2) atol = 1e-12
    @test vec(std(problem.inputs[:, problem.train]; dims = 2)) ≈ ones(2) atol = 1e-12

    for index in problem.train
        label = problem.labels[index]
        @test all(problem.targets[:, index] .>= problem.branch_lower[:, label])
        @test all(problem.targets[:, index] .<= problem.branch_upper[:, label])
    end
end

@testset "Phase 8 analytic boxes match exact backend" begin
    dataset = load_iris(joinpath(@__DIR__, "..", "data", "phase8", "iris.data"))
    problem = prepare_problem(dataset, 2)
    rng = MersenneTwister(20271115)
    for _ in 1:20
        point = 2.5 .* randn(rng, 2)
        analytic = Phase8Iris.analytic_project(problem, point)
        backend = lazy_branch_search(problem.model, point; store_prune_records = false)
        @test backend.status == :optimal
        @test only(backend.choices) == analytic.branch
        @test backend.y ≈ analytic.y atol = 1e-6
        @test backend.objective ≈ analytic.objective atol = 1e-6
        @test backend.report !== nothing && backend.report.feasible
    end
end

@testset "Phase 8 matched initialization and one-step training" begin
    dataset = load_iris(joinpath(@__DIR__, "..", "data", "phase8", "iris.data"))
    problem = prepare_problem(dataset, 3)
    first_network = initialize_network(3)
    second_network = initialize_network(3)
    @test first_network.W1 == second_network.W1
    @test first_network.b1 == second_network.b1
    @test first_network.W2 == second_network.W2
    @test first_network.b2 == second_network.b2

    blackbox = train_method(
        problem, :blackbox, 3;
        steps = 1, batch_size = 2, samples = 2, evaluation_interval = 1,
    )
    sage = train_method(
        problem, :sage, 3;
        steps = 1, batch_size = 2, samples = 2, evaluation_interval = 1,
    )
    @test blackbox.trajectory[1] == sage.trajectory[1]
    @test blackbox.center_global_searches == 2
    @test sage.center_global_searches == 2
    @test blackbox.perturbed_global_searches == 4
    @test 0 <= sage.perturbed_global_searches <= 4
    @test blackbox.logical_failures == 0
    @test sage.logical_failures == 0
    @test sage.unresolved_samples == 0
    @test evaluate_network(sage.network, problem, problem.test).feasibility_rate == 1.0
end
