using Random

@testset "Certified geometric lower bounds" begin
    dm = DisjunctiveModel(1)
    set_bounds!(dm, lower = [0.0], upper = [10.0])
    add_disjunction!(
        dm,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)];
        name = :split,
    )

    yhat = [4.2]
    @test geometric_lower_bound(dm, yhat, [0]) == 0.0
    @test geometric_lower_bound(dm, yhat, [1]) ≈ 0.5 * (4.2 - 2.0)^2
    @test geometric_lower_bound(dm, yhat, [2]) ≈ 0.5 * (5.0 - 4.2)^2

    root = solve_node_relaxation(dm, yhat, [0])
    right = solve_node_relaxation(dm, yhat, [2])
    @test root.status == :optimal
    @test root.primal_objective <= 1e-12
    @test right.status == :optimal
    @test isapprox(right.primal_objective, 0.32; atol = 1e-7)
    @test right.certified_lower_bound ≈ 0.32
    @test right.constraint_violation <= 1e-6

    @test_throws DimensionMismatch geometric_lower_bound(dm, yhat, Int[])
    @test_throws ArgumentError solve_node_relaxation(dm, yhat, [3])

    box_model = DisjunctiveModel(2)
    add_disjunction!(
        box_model,
        [
            LinearConstraint([1.0, 0.0], :>=, 1.0),
            LinearConstraint([0.0, 1.0], :>=, 2.0),
        ],
        [LinearConstraint([1.0, 0.0], :<=, -1.0)],
    )
    # Exact for the selected axis-aligned box: 0.5 * (1^2 + 2^2).
    @test geometric_lower_bound(box_model, [0.0, 0.0], [1]) ≈ 2.5

    coupled = DisjunctiveModel(2)
    add_linear_constraint!(coupled, [1.0, 1.0], :<=, 0.0)
    add_linear_constraint!(coupled, [1.0, -1.0], :<=, 0.0)
    # Each row alone gives objective 0.25 at [1, 0], whereas their
    # nonnegative dual combination certifies the exact objective 0.5.
    @test isapprox(
        geometric_lower_bound(coupled, [1.0, 0.0], Int[]),
        0.5;
        atol = 1e-8,
    )
end

@testset "Dual geometric bounds stay below coupled exact completions" begin
    rng = MersenneTwister(2026111413)
    for _ in 1:10
        model = DisjunctiveModel(2)
        set_bounds!(model, lower = fill(-2.0, 2), upper = fill(2.0, 2))
        for _ in 1:3
            direction = randn(rng, 2)
            direction ./= sqrt(sum(abs2, direction))
            add_disjunction!(
                model,
                [LinearConstraint(direction, :<=, -0.3)],
                [LinearConstraint(direction, :>=, 0.3)],
            )
        end
        point = -1.5 .+ 3.0 .* rand(rng, 2)
        oracle = exhaustive_project(model, point)
        for first in 0:2, second in 0:2, third in 0:2
            assignments = [first, second, third]
            matching = filter(oracle.branch_results) do branch
                branch.accepted && all(
                    assignments[index] == 0 ||
                    assignments[index] == branch.choices[index]
                    for index in eachindex(assignments)
                )
            end
            isempty(matching) && continue
            exact_completion = minimum(branch.objective for branch in matching)
            bound = geometric_lower_bound(model, point, assignments)
            @test bound <= exact_completion + 1e-8
        end
    end
end

@testset "Lazy search resolves the Phase 0 counterexample" begin
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
    search = lazy_branch_search(dm, yhat)
    oracle = exhaustive_project(dm, yhat)
    audit = audit_pruning(search, oracle)

    @test search.status == :optimal
    @test is_certified(search)
    @test search.choices == [2, 1]
    @test isapprox(search.y, [0.75, 0.1]; atol = 1e-7)
    @test search.objective ≈ oracle.objective
    @test search.report.feasible
    @test search.leaf_solves <= 2
    @test search.pruned_branches >= 2
    @test search.leaf_solves + search.pruned_branches == search.total_branches
    @test search.termination_reason == :complete
    @test audit.unsafe_prunes == 0
    @test audit.unsafe_pruned_branches == 0
    @test audit.all_prunes_recorded
    @test audit_passed(audit)

    reversed = lazy_branch_search(dm, yhat; rule_ordering = [:y_split, :x_split])
    @test reversed.rule_order == [2, 1]
    @test reversed.status == :optimal
    @test reversed.objective ≈ oracle.objective
end

@testset "Lazy search reuses a certified external incumbent" begin
    dm = DisjunctiveModel(1)
    add_disjunction!(
        dm,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)],
    )
    point = [3.2]
    reference = project_branch(dm, point, [1])
    reused = lazy_branch_search(
        dm,
        point;
        incumbent_result = reference,
        seed_incumbent = false,
    )
    oracle = exhaustive_project(dm, point)
    @test reused.status == :optimal
    @test reused.choices == oracle.choices
    @test reused.objective ≈ oracle.objective
    @test reused.leaf_solves + reused.pruned_branches == reused.total_branches
    stale = project_branch(dm, [1.0], [1])
    @test_throws ArgumentError lazy_branch_search(
        dm,
        point;
        incumbent_result = stale,
    )
end

@testset "Lazy search preserves deterministic ties" begin
    dm = DisjunctiveModel(1)
    add_disjunction!(
        dm,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)],
    )
    search = lazy_branch_search(dm, [3.5])
    oracle = exhaustive_project(dm, [3.5])
    audit = audit_pruning(search, oracle)

    @test search.status == :optimal
    @test search.choices == [1]
    @test search.y ≈ [2.0]
    @test search.leaf_solves == 2
    @test search.pruned_branches == 0
    @test audit_passed(audit)
end

@testset "Lazy search matches seeded random exact oracles" begin
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

        for _ in 1:10
            yhat = -1.5 .+ 3.0 .* rand(rng, dimension)
            search = lazy_branch_search(dm, yhat)
            oracle = exhaustive_project(dm, yhat)
            audit = audit_pruning(search, oracle)

            @test search.status == oracle.status == :optimal
            @test isapprox(search.objective, oracle.objective; atol = 1e-7)
            @test search.report.feasible
            @test search.leaf_solves + search.pruned_branches == search.total_branches
            @test audit.unsafe_prunes == 0
            @test audit_passed(audit)
        end
    end
end

@testset "Controlled instance prunes more than 99 percent of branches" begin
    dimension = 8
    dm = DisjunctiveModel(dimension)
    set_bounds!(dm, lower = zeros(dimension), upper = ones(dimension))
    for coordinate in 1:dimension
        direction = zeros(dimension)
        direction[coordinate] = 1.0
        add_disjunction!(
            dm,
            [LinearConstraint(direction, :<=, 0.2)],
            [LinearConstraint(direction, :>=, 0.8)],
        )
    end

    yhat = fill(0.9, dimension)
    search = lazy_branch_search(dm, yhat)
    oracle = exhaustive_project(dm, yhat)
    audit = audit_pruning(search, oracle)
    prune_fraction = search.pruned_branches / search.total_branches

    @test search.status == :optimal
    @test search.objective <= 1e-12
    @test search.choices == fill(2, dimension)
    @test search.total_branches == 256
    @test search.leaf_solves == 1
    @test search.pruned_branches == 255
    @test prune_fraction > 0.99
    @test search.relaxation_solves == 8
    @test search.explored_nodes == 8
    @test audit.unsafe_prunes == 0
    @test audit_passed(audit)
end

@testset "Lazy search infeasibility, convex case, and node limit" begin
    impossible = DisjunctiveModel(1)
    add_linear_constraint!(impossible, [1.0], :<=, 0.0)
    add_linear_constraint!(impossible, [1.0], :>=, 1.0)
    no_solution = lazy_branch_search(impossible, [0.5])
    @test no_solution.status == :infeasible
    @test is_certified(no_solution)
    @test no_solution.leaf_solves == 1
    @test no_solution.infeasible_leaves == 1

    impossible_rules = DisjunctiveModel(1)
    add_linear_constraint!(impossible_rules, [1.0], :<=, 0.0)
    add_linear_constraint!(impossible_rules, [1.0], :>=, 1.0)
    add_disjunction!(
        impossible_rules,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, -2.0)],
    )
    root_infeasible = lazy_branch_search(impossible_rules, [0.5])
    @test root_infeasible.status == :infeasible
    @test root_infeasible.pruned_branches == 2
    @test root_infeasible.infeasibility_prunes == 1

    convex = DisjunctiveModel(1)
    set_bounds!(convex, lower = [0.0], upper = [1.0])
    ordinary = lazy_branch_search(convex, [2.0])
    @test ordinary.status == :optimal
    @test ordinary.y ≈ [1.0]
    @test ordinary.leaf_solves == 1
    @test ordinary.total_branches == 1

    limited_model = DisjunctiveModel(2)
    for coordinate in 1:2
        direction = zeros(2)
        direction[coordinate] = 1.0
        add_disjunction!(
            limited_model,
            [LinearConstraint(direction, :<=, 0.2)],
            [LinearConstraint(direction, :>=, 0.8)],
        )
    end
    limited = lazy_branch_search(
        limited_model,
        [0.5, 0.5];
        max_nodes = 1,
        seed_incumbent = false,
    )
    @test limited.status == :uncertified
    @test !is_certified(limited)
    @test limited.termination_reason == :node_limit

    @test_throws ArgumentError lazy_branch_search(limited_model, [0.5, 0.5]; max_nodes = 0)
    @test_throws ArgumentError lazy_branch_search(
        limited_model,
        [0.5, 0.5];
        rule_ordering = [1, 1],
    )

    unauditable = lazy_branch_search(
        limited_model,
        [0.9, 0.9];
        store_prune_records = false,
    )
    unauditable_oracle = exhaustive_project(limited_model, [0.9, 0.9])
    incomplete_audit = audit_pruning(unauditable, unauditable_oracle)
    @test unauditable.pruned_nodes > 0
    @test !incomplete_audit.all_prunes_recorded
    @test !audit_passed(incomplete_audit)
end

