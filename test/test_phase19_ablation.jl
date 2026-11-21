if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
if !isdefined(@__MODULE__, :Phase18External)
    include(joinpath(@__DIR__, "..", "scripts", "phase18_external_common.jl"))
end
if !isdefined(@__MODULE__, :Phase19Ablation)
    include(joinpath(@__DIR__, "..", "scripts", "phase19_ablation_common.jl"))
end

using Random

@testset "Phase 19 one-factor component ablation" begin
    P19 = Phase19Ablation
    problem, network = Phase18External.setup_problem(1; small = true)
    context = P19.center_context(problem, network, 1)
    samples = sample_antithetic_ball_smoothing(
        MersenneTwister(19), length(context.point), 2; beta = 2.0,
    )
    estimates = Dict(variant => P19.run_variant(
        variant,
        context,
        samples,
        0.5;
        rng = MersenneTwister(100 + index),
        parallel = true,
    ) for (index, variant) in enumerate(P19.PHASE19_VARIANTS))

    @test Set(keys(estimates)) == Set(P19.PHASE19_VARIANTS)
    for estimate in values(estimates)
        @test length(estimate.gradient) == length(context.point)
        @test all(isfinite, estimate.gradient)
        @test estimate.unresolved == 0
        @test estimate.logical_failures == 0
        @test estimate.global_searches <= 2
        @test estimate.branch_solves >= 0
        @test estimate.workers <= Threads.nthreads()
    end
    @test estimates[:compact_blackbox].global_searches == 2
    @test estimates[:residual_decomposition].global_searches == 2
    @test estimates[:compact_blackbox].ess == 2.0

    audit = P19.ablate_task(
        context,
        7;
        radius = 0.5,
        estimator_samples = 2,
        repetitions = 2,
        reference_samples = 4,
        parallel = true,
    )
    @test length(audit.raw) == 2length(P19.PHASE19_VARIANTS)
    @test length(audit.summaries) == length(P19.PHASE19_VARIANTS)
    @test all(isfinite(row.relative_bias) for row in audit.summaries)
    @test all(isfinite(row.rmse) for row in audit.summaries)
    @test isnan(P19._finite_mean(Float64[]))
    @test isnan(P19._finite_mean((NaN for _ in 1:3)))
    @test P19._finite_mean([1.0, NaN, 3.0]) == 2.0

    # Regression for the accounting bug found while designing Phase 19: SAGE
    # lazy leaves are competitor solves, not a second set of complete QPs.
    trained = Phase10JobShop.train_method(
        problem,
        network,
        :sage_multiscale,
        2;
        epochs = 1,
        max_steps = 1,
        samples = 2,
        radius = 0.5,
        learning_rate = 1e-3,
        evaluation_splits = :none,
        parallel_sage_samples = true,
    )
    @test trained.complete_branch_solves == trained.center_leaf_solves +
        trained.reference_branch_solves + trained.competitor_branch_solves +
        trained.geometry_branch_solves + trained.lazy_leaf_solves
    @test trained.lazy_leaf_solves == 0
end
