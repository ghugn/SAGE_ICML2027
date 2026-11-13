using Random

@testset "Exact branch projection: analytic intervals" begin
    dm = DisjunctiveModel(1)
    set_bounds!(dm, lower = [0.0], upper = [10.0])
    add_disjunction!(
        dm,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)];
        name = :split,
    )

    @test num_branches(dm) == 2

    right = exact_project(dm, [4.2])
    @test right.status == :optimal
    @test is_certified(right)
    @test right.choices == [2]
    @test right.y ≈ [5.0]
    @test right.objective ≈ 0.5 * (5.0 - 4.2)^2
    @test right.distance ≈ 0.8
    @test right.report.feasible
    @test right.total_branches == 2
    @test right.solved_branches == 2
    @test right.feasible_branches == 2
    @test right.infeasible_branches == 0
    @test right.unresolved_branches == 0

    inside = exhaustive_project(dm, [1.2])
    @test inside.choices == [1]
    @test isapprox(inside.y, [1.2]; atol = 1e-6)
    @test inside.objective <= 1e-12

    # Both intervals are at distance 1.5; lexicographic branch 1 must win.
    tie = exhaustive_project(dm, [3.5])
    @test tie.choices == [1]
    @test tie.y ≈ [2.0]
    @test tie.objective ≈ 1.125

    branch = project_branch(dm, [4.2], [2])
    @test branch.accepted
    @test !branch.proven_infeasible
    @test branch.solver_status == MOI.OPTIMAL
    @test branch.report.feasible
    @test branch.selected_branch_violation <= 1e-6

    @test_throws DimensionMismatch project_branch(dm, [4.2], Int[])
    @test_throws DimensionMismatch project_branch(dm, [4.2, 1.0], [1])
    @test_throws ArgumentError project_branch(dm, [4.2], [0])
    @test_throws ArgumentError project_branch(dm, [NaN], [1])
    @test_throws ArgumentError exhaustive_project(dm, [4.2]; tie_tol = -1.0)
end

@testset "Exact oracle matches random separable analytic projections" begin
    rng = MersenneTwister(20261113)

    for dimension in (2, 3)
        dm = DisjunctiveModel(dimension)
        set_bounds!(dm, lower = fill(-2.0, dimension), upper = fill(2.0, dimension))

        for coordinate in 1:dimension
            direction = zeros(dimension)
            direction[coordinate] = 1.0
            add_disjunction!(
                dm,
                [LinearConstraint(direction, :<=, -0.4)],
                [LinearConstraint(direction, :>=, 0.6)],
            )
        end

        for _ in 1:12
            yhat = -1.5 .+ 3.0 .* rand(rng, dimension)
            expected = map(yhat) do value
                if value <= -0.4 || value >= 0.6
                    value
                elseif value <= 0.1
                    -0.4
                else
                    0.6
                end
            end

            result = exhaustive_project(dm, yhat)
            @test result.status == :optimal
            @test isapprox(result.y, expected; atol = 1e-6)
            @test isapprox(
                result.objective,
                0.5 * sum(abs2, expected .- yhat);
                atol = 1e-7,
            )
            @test result.report.feasible
        end
    end
end

@testset "Exact oracle: deterministic multidimensional tie" begin
    dm = DisjunctiveModel(2)
    set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])
    add_disjunction!(
        dm,
        [LinearConstraint([1.0, 0.0], :<=, 0.2)],
        [LinearConstraint([1.0, 0.0], :>=, 0.8)],
    )
    add_disjunction!(
        dm,
        [LinearConstraint([0.0, 1.0], :<=, 0.2)],
        [LinearConstraint([0.0, 1.0], :>=, 0.8)],
    )

    result = exhaustive_project(dm, [0.5, 0.5])
    @test result.status == :optimal
    @test result.total_branches == 4
    @test result.choices == [1, 1]
    @test isapprox(result.y, [0.2, 0.2]; atol = 1e-7)
    @test result.objective ≈ 0.09
    @test result.report.feasible
end

@testset "Exact oracle resolves Phase 0 relaxation counterexample" begin
    dm = DisjunctiveModel(2)
    set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])
    add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)
    add_disjunction!(
        dm,
        [LinearConstraint([1.0, 0.0], :<=, 0.25)],
        [LinearConstraint([1.0, 0.0], :>=, 0.75)];
        name = :x_split,
    )
    add_disjunction!(
        dm,
        [LinearConstraint([0.0, 1.0], :<=, 0.25)],
        [LinearConstraint([0.0, 1.0], :>=, 0.75)];
        name = :y_split,
    )

    yhat = [0.5, 0.1]
    exact = exhaustive_project(dm, yhat)
    relaxed = project(dm, yhat; formulation = :dnf)

    @test exact.status == :optimal
    @test exact.choices == [2, 1]
    @test isapprox(exact.y, [0.75, 0.1]; atol = 1e-7)
    @test exact.objective ≈ 0.03125
    @test exact.feasible_branches == 3
    @test exact.infeasible_branches == 1
    @test exact.report.feasible
    @test is_logically_feasible(dm, exact.y)
    @test !is_logically_feasible(dm, relaxed.y)
    @test sum(abs2, relaxed.y .- yhat) < sum(abs2, exact.y .- yhat)
end

@testset "Exact oracle: infeasibility and ordinary convex model" begin
    partially_infeasible = DisjunctiveModel(1)
    set_bounds!(partially_infeasible, lower = [0.0], upper = [10.0])
    add_linear_constraint!(partially_infeasible, [1.0], :>=, 4.0)
    add_disjunction!(
        partially_infeasible,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)],
    )

    partial = exhaustive_project(partially_infeasible, [3.0])
    @test partial.status == :optimal
    @test partial.choices == [2]
    @test partial.y ≈ [5.0]
    @test partial.feasible_branches == 1
    @test partial.infeasible_branches == 1
    @test partial.branch_results[1].proven_infeasible

    impossible = DisjunctiveModel(1)
    add_linear_constraint!(impossible, [1.0], :<=, 0.0)
    add_linear_constraint!(impossible, [1.0], :>=, 1.0)
    no_solution = exhaustive_project(impossible, [0.5])
    @test no_solution.status == :infeasible
    @test is_certified(no_solution)
    @test isempty(no_solution.y)
    @test no_solution.total_branches == 1
    @test no_solution.infeasible_branches == 1

    convex = DisjunctiveModel(1)
    set_bounds!(convex, lower = [0.0], upper = [1.0])
    ordinary = exhaustive_project(convex, [2.0]; store_branch_results = false)
    @test ordinary.status == :optimal
    @test ordinary.y ≈ [1.0]
    @test isempty(ordinary.choices)
    @test ordinary.total_branches == 1
    @test ordinary.objective ≈ 0.5
    @test isempty(ordinary.branch_results)
end

@testset "Exact oracle branch-count safety guard" begin
    dm = DisjunctiveModel(1)
    for rule_index in 1:4
        add_disjunction!(
            dm,
            [LinearConstraint([1.0], :<=, 1.0 + rule_index)],
            [LinearConstraint([1.0], :>=, -1.0 - rule_index)],
            [LinearConstraint([1.0], :<=, 2.0 + rule_index)],
        )
    end
    @test num_branches(dm) == 81
    @test_throws ArgumentError exhaustive_project(dm, [0.0]; max_branches = 80)
    @test_throws ArgumentError exhaustive_project(dm, [0.0]; max_branches = 0)
end

