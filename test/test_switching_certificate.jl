using Random

function _assert_no_grid_switch(
    model::DisjunctiveModel,
    center::Vector{Float64},
    certificate::SwitchingCertificate,
    offsets,
)
    @test certifies_stability(certificate)
    for offset in offsets
        delta = offset isa Number ? [Float64(offset)] : Float64.(collect(offset))
        @test sqrt(sum(abs2, delta)) <= certificate.radius + 1e-12
        perturbed = exhaustive_project(model, center .+ delta)
        @test perturbed.status == :optimal
        @test perturbed.choices == certificate.winner
    end
end

@testset "Switching certificate: analytic interval and strict ties" begin
    dm = DisjunctiveModel(1)
    add_disjunction!(
        dm,
        [LinearConstraint([1.0], :<=, 2.0)],
        [LinearConstraint([1.0], :>=, 5.0)];
        name = :split,
    )

    stable = switching_certificate(dm, [1.0], 0.5)
    @test stable.status == :stable
    @test is_certified(stable)
    @test stable.winner == [1]
    @test stable.total_competitor_branches == 1
    @test stable.stable_pruned_branches == 1
    @test stable.competitive_branches == 0
    @test isempty(competitor_set(stable))
    @test stable.groups[1].lower_distance ≈ 4.0
    @test isapprox(stable.groups[1].distance_margin, 4.0; atol = 1e-6)
    _assert_no_grid_switch(dm, [1.0], stable, range(-0.5, 0.5; length = 41))

    wide = switching_certificate(dm, [1.0], 2.6)
    @test wide.status == :not_certified
    @test !certifies_stability(wide)
    @test wide.competitive_branches == 1
    @test length(competitor_set(wide)) == 1
    @test exhaustive_project(dm, [3.6]).choices == [2]
    local_radius = certified_local_radius(wide)
    @test local_radius > 0.0
    @test wide.minimum_margin > 2local_radius +
        max(wide.required_margin - 2wide.radius, 0.0)
    local_certificate = switching_certificate(dm, [1.0], local_radius)
    @test certifies_stability(local_certificate)

    tie = switching_certificate(dm, [3.5], 0.0)
    @test tie.status == :not_certified
    @test tie.winner == [1]
    @test tie.minimum_margin ≈ 0.0 atol = 1e-7
    @test certified_local_radius(tie) == 0.0

    @test_throws ArgumentError switching_certificate(dm, [1.0], -0.1)
    @test_throws ArgumentError switching_certificate(dm, [1.0], Inf)
    @test_throws DimensionMismatch switching_certificate(dm, [1.0, 2.0], 0.1)
    @test_throws ArgumentError certified_local_radius(wide; safety_factor = 0.5)
    stale_center = exhaustive_project(dm, [1.0])
    @test_throws ArgumentError switching_certificate(
        dm,
        [1.2],
        0.1;
        center_result = stale_center,
    )
end

@testset "Competitor partition is complete, disjoint by prefix, and ordered" begin
    dm = DisjunctiveModel(3)
    for coordinate in 1:3
        direction = zeros(3)
        direction[coordinate] = 1.0
        add_disjunction!(
            dm,
            [LinearConstraint(direction, :<=, -0.2)],
            [LinearConstraint(direction, :>=, 0.2)];
            name = Symbol("r$(coordinate)"),
        )
    end

    groups = competitor_partition(dm, [2, 2, 2])
    @test length(groups) == 3
    @test sum(group.branch_count for group in groups) == 7
    @test [group.branch_count for group in groups] == [4, 2, 1]
    @test groups[1].assignments == [1, 0, 0]
    @test groups[2].assignments == [2, 1, 0]
    @test groups[3].assignments == [2, 2, 1]

    reversed = competitor_partition(dm, [2, 2, 2]; rule_ordering = [3, 2, 1])
    @test sum(group.branch_count for group in reversed) == 7
    @test reversed[1].assignments == [0, 0, 1]
    @test reversed[3].assignments == [1, 2, 2]
end

@testset "2D exhaustive perturbation grid has zero false stable certificates" begin
    dm = DisjunctiveModel(2)
    for coordinate in 1:2
        direction = zeros(2)
        direction[coordinate] = 1.0
        add_disjunction!(
            dm,
            [LinearConstraint(direction, :<=, -0.4)],
            [LinearConstraint(direction, :>=, 0.6)],
        )
    end

    center = [-1.0, -0.9]
    radius = 0.2
    certificate = switching_certificate(dm, center, radius)
    offsets = Tuple{Float64, Float64}[]
    for dx in range(-radius, radius; length = 13)
        for dy in range(-radius, radius; length = 13)
            dx^2 + dy^2 <= radius^2 + 1e-12 && push!(offsets, (dx, dy))
        end
    end
    _assert_no_grid_switch(dm, center, certificate, offsets)
end

@testset "Random near-boundary audit is conservative" begin
    rng = MersenneTwister(20261113)
    dm = DisjunctiveModel(1)
    add_disjunction!(
        dm,
        [LinearConstraint([1.0], :<=, -0.25)],
        [LinearConstraint([1.0], :>=, 0.35)],
    )

    stable_claims = 0
    checked_points = 0
    for _ in 1:40
        center = [-1.0 + 2.0 * rand(rng)]
        radius = 0.25 * rand(rng)
        certificate = switching_certificate(dm, center, radius)
        if certifies_stability(certificate)
            stable_claims += 1
            for delta in range(-radius, radius; length = 21)
                result = exhaustive_project(dm, center .+ delta)
                @test result.status == :optimal
                @test result.choices == certificate.winner
                checked_points += 1
            end
        end
    end
    @test stable_claims > 0
    @test checked_points > 0
end

@testset "Overlaps, no-disjunction models, and unresolved centers" begin
    overlap = DisjunctiveModel(1)
    add_disjunction!(
        overlap,
        [LinearConstraint([1.0], :<=, 1.0)],
        [LinearConstraint([1.0], :>=, 0.0)],
    )
    ambiguous = switching_certificate(overlap, [0.5], 0.0)
    @test ambiguous.status == :not_certified
    @test ambiguous.competitive_branches == 1

    convex = DisjunctiveModel(1)
    set_bounds!(convex, lower = [0.0], upper = [1.0])
    single = switching_certificate(convex, [2.0], 100.0)
    @test single.status == :stable
    @test single.total_competitor_branches == 0
    @test single.minimum_margin == Inf

    impossible = DisjunctiveModel(1)
    add_linear_constraint!(impossible, [1.0], :<=, 0.0)
    add_linear_constraint!(impossible, [1.0], :>=, 1.0)
    absent = switching_certificate(impossible, [0.5], 0.1)
    @test absent.status == :center_infeasible
    @test !certifies_stability(absent)

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
    unresolved = switching_certificate(
        limited_model,
        [0.5, 0.5],
        0.1;
        center_result = limited,
    )
    @test unresolved.status == :center_uncertified
    @test !certifies_stability(unresolved)
end

