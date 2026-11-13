@testset "Original logical feasibility" begin
    dm = DisjunctiveModel(2)
    set_bounds!(dm, lower = [0.0, 0.0], upper = [1.0, 1.0])
    add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)
    add_disjunction!(
        dm,
        [LinearConstraint([1.0, 0.0], :<=, 0.25)],
        [LinearConstraint([1.0, 0.0], :>=, 0.75)];
        name = :x_split,
    )

    left = check_feasibility(dm, [0.2, 0.6])
    @test left.feasible
    @test left.max_violation <= left.tolerance
    @test left.selected_disjuncts == [1]

    right = check_feasibility(dm, [0.8, 0.0])
    @test right.feasible
    @test right.selected_disjuncts == [2]
    @test is_logically_feasible(dm, [0.8, 0.0])

    logical_gap = check_feasibility(dm, [0.5, 0.3])
    @test !logical_gap.feasible
    @test isapprox(logical_gap.disjunction_violations[1], 0.25; atol = 1e-12)
    @test logical_gap.bound_violation == 0.0
    @test logical_gap.global_violation == 0.0

    bound_gap = check_feasibility(dm, [1.1, 0.0])
    @test !bound_gap.feasible
    @test isapprox(bound_gap.bound_violation, 0.1; atol = 1e-12)

    nonfinite = check_feasibility(dm, [NaN, 1.0])
    @test !nonfinite.feasible
    @test nonfinite.max_violation == Inf

    @test_throws DimensionMismatch check_feasibility(dm, [0.5])
    @test_throws ArgumentError check_feasibility(dm, [0.5, 0.5]; atol = -1e-6)

    # An OPTIMAL relaxation solution is not automatically a logical certificate.
    relaxed = project(dm, [0.5, 0.1])
    @test relaxed.status == MOI.OPTIMAL
    relaxed_report = check_feasibility(dm, relaxed.y)
    @test !relaxed_report.feasible
    @test relaxed_report.disjunction_violations[1] > 0.0
end

@testset "Constraint residuals" begin
    @test constraint_violation(LinearConstraint([1.0], :<=, 1.0), [1.2]) ≈ 0.2
    @test constraint_violation(LinearConstraint([1.0], :>=, 1.0), [0.7]) ≈ 0.3
    @test constraint_violation(LinearConstraint([2.0], :(==), 1.0), [0.6]) ≈ 0.2
    @test_throws DimensionMismatch constraint_violation(
        LinearConstraint([1.0, 0.0], :<=, 1.0),
        [0.5],
    )
end

