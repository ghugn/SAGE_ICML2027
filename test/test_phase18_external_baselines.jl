if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
if !isdefined(@__MODULE__, :Phase18External)
    include(joinpath(@__DIR__, "..", "scripts", "phase18_external_common.jl"))
end
using LinearAlgebra
using Random
using Statistics

@testset "Phase 18 external baseline formula and contracts" begin
    EB = Phase18External
    JS = Phase10JobShop
    for label in (:hard_local, :ste, :compact_blackbox, :sage_n4, :dbb_projection, :dpo_gaussian)
        candidates = EB.candidate_specs(label)
        @test length(candidates) == 9
        @test length(unique(candidates)) == 9
        @test EB.default_spec(label) in candidates
        @test all(candidate.samples == 4 for candidate in candidates)
    end
    @test sort(unique(spec.lambda for spec in EB.candidate_specs(:dbb_projection))) == [0.05, 0.5, 5.0]
    @test sort(unique(spec.gaussian_scale for spec in EB.candidate_specs(:dpo_gaussian))) == [0.5, 1.0, 2.0]
    @test sort(unique(spec.radius for spec in EB.candidate_specs(:sage_n4))) == [0.25, 0.5, 1.0]
    point = [0.2, -0.3]
    upstream = [0.7, -0.4]
    @test EB.dbb_projection_vjp(identity, point, upstream; lambda = 0.5) ≈ upstream
    projection(z) = [z[1], 0.0]
    @test EB.dbb_projection_vjp(projection, point, upstream; lambda = 0.2) ≈ [0.7, 0.0]
    @test_throws ArgumentError EB.dbb_projection_vjp(identity, point, upstream; lambda = 0)
    @test_throws ArgumentError EB.dbb_projection_vjp(identity, point, upstream; lambda = Inf)
    @test_throws DimensionMismatch EB.dbb_projection_vjp(identity, point, [1.0]; lambda = 1)

    # Exact finite candidate set independently checks the objective-tilt
    # identity, including the minus sign at the network prediction z.
    candidates = [[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]]
    project(z) = candidates[argmin([sum(abs2, y - z) for y in candidates])]
    lambda = 0.8
    tilted = candidates[argmin([0.5sum(abs2, y - point) + lambda * dot(upstream, y)
        for y in candidates])]
    @test project(point - lambda * upstream) == tilted
    @test EB.dbb_projection_vjp(project, point, upstream; lambda = lambda) ==
        (project(point) - tilted) / lambda

    # Deterministic directions with exact identity covariance make the Gaussian
    # identity-map score estimate equal v without a stochastic tolerance.
    directions = sqrt(2.0) .* [1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0]
    estimate = EB.gaussian_perturbed_vjp(identity, point, upstream, directions; sigma = 0.1)
    @test estimate.gradient ≈ upstream
    serial = EB.gaussian_perturbed_vjp(projection, point, upstream, directions; sigma = 0.3)
    threaded = EB.gaussian_perturbed_vjp(projection, point, upstream, directions;
        sigma = 0.3, parallel = true)
    @test serial.gradient ≈ [0.7, 0.0]
    @test serial.gradient == threaded.gradient
    @test threaded.workers == min(Threads.nthreads(), size(directions, 2))
    @test_throws ArgumentError EB.gaussian_perturbed_vjp(identity, point, upstream,
        directions; sigma = -1)

    # Nonlinear map: independent high-precision finite-difference integral
    # checks the score VJP against d/dz E[tanh(z+sigma*epsilon)].
    xs = collect(range(-8.0, 8.0; length = 40001))
    dx = xs[2] - xs[1]
    weights = exp.(-xs .^ 2 ./ 2) ./ sqrt(2pi) .* dx
    z, sigma = 0.23, 0.4
    score_gradient = sum(weights .* xs .* tanh.(z .+ sigma .* xs)) / sigma
    smooth(t) = sum(weights .* tanh.(t .+ sigma .* xs))
    finite_difference = (smooth(z + 1e-5) - smooth(z - 1e-5)) / 2e-5
    @test score_gradient ≈ finite_difference atol = 1e-8

    problem, pretrained = EB.setup_problem(1; small = true)
    @test length(problem.iid.instances) == 4
    @test problem.iid.raw_inputs == problem.ood.raw_inputs # validation-only alias
    @test problem.input_mean ≈ vec(mean(problem.train.raw_inputs; dims = 2))
    heldout, _ = EB.setup_problem(1; pool = :heldout, small = true)
    @test heldout.iid.raw_inputs != problem.iid.raw_inputs
    @test heldout.ood.raw_inputs != problem.ood.raw_inputs
    @test heldout.train.raw_inputs == problem.train.raw_inputs
    for label in (:dbb_projection, :dpo_gaussian, :compact_blackbox, :sage_n4)
        spec = EB.default_spec(label)
        states = Any[]
        uninterrupted = EB.train_external(problem, pretrained, spec, 1; steps = 2,
            state_checkpoint_interval = 1,
            state_checkpoint_callback = state -> push!(states, state))
        resumed = EB.train_external(problem, pretrained, spec, 1; steps = 2,
            resume_state = states[1])
        serial_run = EB.train_external(problem, pretrained, spec, 1;
            steps = 2, parallel = false)
        @test EB.network_equal(uninterrupted.network, resumed.network)
        @test EB.network_equal(uninterrupted.network, serial_run.network)
        @test uninterrupted.perturbed_global_searches == resumed.perturbed_global_searches
        @test uninterrupted.complete_branch_solves == serial_run.complete_branch_solves
        @test uninterrupted.logical_failures == 0
        @test uninterrupted.unresolved_samples == 0
        @test all(isnan(row.ood_task_loss) for row in uninterrupted.trajectory)
        expected_searches = label == :dbb_projection ? 2 : label == :sage_n4 ? nothing : 8
        expected_searches === nothing ||
            @test uninterrupted.perturbed_global_searches == expected_searches
        evaluated = JS.evaluate_network(uninterrupted.network, problem.iid; config = problem.config)
        @test evaluated.csat == 1.0
    end

    # A zero-work time stop must return actual completed steps, not max_steps.
    stopped = EB.train_external(problem, pretrained, EB.default_spec(:ste), 1;
        steps = 4, wall_time_limit_seconds = eps())
    @test stopped.steps == 0
    @test stopped.stop_reason == :wall_time
    @test EB.network_equal(stopped.network, pretrained)

    # Existing default training behavior and the new hook agree exactly.
    original = JS.train_method(problem, pretrained, :ste, 1;
        epochs = 1, max_steps = 2, samples = 4, radius = 0.5, learning_rate = 1e-3,
        evaluation_interval = 64, evaluation_splits = :none)
    hooked = EB.train_external(problem, pretrained, EB.default_spec(:ste), 1; steps = 2)
    @test EB.network_equal(original.network, hooked.network)
    @test original.complete_branch_solves == hooked.complete_branch_solves
    @test_throws ArgumentError EB.train_external(problem, pretrained,
        merge(EB.default_spec(:dpo_gaussian), (gaussian_scale = 0.0,)), 1; steps = 1)
end
