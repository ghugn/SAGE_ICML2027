module Phase18RunnerHarness
include(joinpath(@__DIR__, "..", "scripts", "benchmark_phase18_external_baselines.jl"))
end

using Test

@testset "Phase 18 runner persistence and statistical reporting" begin
    runner = Phase18RunnerHarness
    @test runner.json((finite = 1.0, missing = NaN, text = "a\"b\nc")) ==
        "{\"finite\":1.0,\"missing\":null,\"text\":\"a\\\"b\\nc\"}"
    @test length(runner.source_fingerprint()) == 64
    @test runner.source_fingerprint() == runner.source_fingerprint()
    @test_throws ErrorException runner.interval([1.0, 2.0])
    @test runner.interval(fill(-1.0, 5)).upper == -1.0
    kaggle_rates = Dict("hard_local" => 0.083, "ste" => 0.0723,
        "compact_blackbox" => 0.1947, "dbb_projection" => 0.1615,
        "dpo_gaussian" => 0.1859, "sage_n4" => 0.2529)
    staged = runner.projected_stage_seconds(kaggle_rates)
    @test sum(values(staged)) / 3600 > 4.0
    @test maximum(values(staged)) / 3600 < 4.0
    @test staged.confirm_b > staged.confirm_a

    mktempdir() do directory
        problem, pretrained = runner.EB.setup_problem(1; small = true)
        for label in (:dbb_projection, :dpo_gaussian)
            spec = runner.EB.default_spec(label)
            row = runner.execute_method(directory, problem, pretrained, spec, 1;
                steps = 2, tag = "tune2")
            @test row.csat == 1.0
            @test isnan(row.ood_loss)
            cached = runner.execute_method(directory, problem, pretrained, spec, 1;
                steps = 2, tag = "tune2")
            @test isequal(row, cached)
            @test isfile(joinpath(directory, "training_states", "seed_1_$(label)_tune2.jls"))
            @test_throws ErrorException runner.execute_method(directory, problem, pretrained,
                merge(spec, (learning_rate = 0.02,)), 1; steps = 2, tag = "tune2")
        end
        rows = NamedTuple[]
        for seed in 11:15
            for label in runner.PHASE18_LABELS
                push!(rows, (seed = seed, method = string(label), tag = "fixed",
                    iid_loss = 0.1, ood_loss = label == :sage_n4 ? 0.1 : 0.2,
                    csat = 1.0, maximum_violation = 0.0, seconds = 10.0,
                    budget_seconds = Inf, stop_reason = "max_steps",
                    perturbed_searches = 4, branch_solves = 8))
            end
            for label in (:ste, :dbb_projection, :dpo_gaussian)
                push!(rows, (seed = seed, method = string(label), tag = "matched_time",
                    iid_loss = 0.1, ood_loss = 0.2, csat = 1.0, maximum_violation = 0.0,
                    seconds = 10.2, budget_seconds = 10.0, stop_reason = "wall_time",
                    perturbed_searches = 4, branch_solves = 8))
            end
        end
        summary = runner.report(directory, rows)
        @test summary.status == "COMPLETE"
        @test summary.safety_gate
        @test summary.matched_time_overshoot_gate
        @test length(summary.comparisons) == 2
        @test all(value.superiority_gate for value in values(summary.comparisons))
        @test length(summary.matched_time_exploratory_ci95) == 3
        @test summary.aggregates["sage_n4/fixed"].runs == 5
        @test isfile(joinpath(directory, "phase18_runs.csv"))
        @test isfile(joinpath(directory, "phase18_summary.json"))
        partial = runner.report(directory, rows[1:3])
        @test partial.status == "PARTIAL"
        @test isempty(partial.comparisons)
    end

    mktempdir() do directory
        spec = runner.EB.default_spec(:sage_n4)
        signature = (schema = runner.PHASE18_SCHEMA, source = runner.source_fingerprint(),
            seed = 11, spec = spec, steps = 1024, tag = "fixed", wall_seconds = Inf)
        row = (seed = 11, method = "sage_n4", tag = "fixed")
        cache = joinpath(directory, "completed_methods", "seed_11_sage_n4_fixed.jls")
        runner.save_atomic(cache, (signature = signature, row = row))
        @test runner.collect_confirmatory_rows(directory) == [row]
        runner.save_atomic(joinpath(directory, "completed_methods", "duplicate.jls"),
            (signature = signature, row = row))
        @test_throws ErrorException runner.collect_confirmatory_rows(directory)
    end

    # Exercise the complete selection matrix without touching held-out data.
    # Adversarial OOD values are rejected, not used as a hidden tie-breaker.
    validation = NamedTuple[]
    for label in runner.PHASE18_LABELS, (index, spec) in enumerate(runner.EB.candidate_specs(label)), seed in (1, 2)
        push!(validation, (seed = seed, method = string(label), tag = "tune$(index)",
            steps = 128, learning_rate = spec.learning_rate, radius = spec.radius,
            lambda = spec.lambda, gaussian_scale = spec.gaussian_scale,
            samples = spec.samples, csat = 1.0, iid_loss = abs(index - 7) + seed * 0.001,
            ood_loss = NaN))
    end
    selected, scores = runner.select_candidates(validation)
    @test length(validation) == 108
    @test all(value.winner == 7 for value in values(scores))
    @test all(selected[label] == runner.EB.candidate_specs(label)[7] for label in runner.PHASE18_LABELS)
    @test_throws ErrorException runner.select_candidates(validation[2:end])
    duplicated = vcat(validation, validation[1:1])
    @test_throws ErrorException runner.select_candidates(duplicated)
    contaminated = copy(validation)
    contaminated[1] = merge(contaminated[1], (ood_loss = -1000.0,))
    @test_throws ErrorException runner.select_candidates(contaminated)
    malformed = copy(validation)
    malformed[1] = merge(malformed[1], (gaussian_scale = 999.0,))
    @test_throws ErrorException runner.select_candidates(malformed)
    tied = [merge(row, (iid_loss = 0.2,)) for row in validation]
    _, tied_scores = runner.select_candidates(tied)
    @test all(value.winner == 1 for value in values(tied_scores))
    pilot_rows = [(method = string(label), steps = 256, tag = "pilot", csat = 1.0,
        iid_loss = 0.1, ood_loss = NaN, seconds = 25.6) for label in runner.PHASE18_LABELS]
    @test runner.remaining_hours(pilot_rows) ≈ 2.0 * 5 * 1024 * 0.9 / 3600
    @test runner.remaining_hours(pilot_rows; seeds = 11:12) ≈
        2.0 * 2 * 1024 * 0.9 / 3600
    @test runner.remaining_hours(pilot_rows; seeds = 13:15) ≈
        2.0 * 3 * 1024 * 0.9 / 3600
    @test_throws ErrorException runner.remaining_hours(pilot_rows; seeds = Int[])
    @test_throws ErrorException runner.remaining_hours(pilot_rows; seeds = [11, 11])
    @test_throws ErrorException runner.remaining_hours(pilot_rows; seeds = [99])
    @test_throws ErrorException runner.remaining_hours(pilot_rows[2:end])
    @test_throws ErrorException runner.remaining_hours(vcat(pilot_rows, pilot_rows[1:1]))
    contaminated_pilot = copy(pilot_rows)
    contaminated_pilot[1] = merge(contaminated_pilot[1], (ood_loss = 0.0,))
    @test_throws ErrorException runner.remaining_hours(contaminated_pilot)
end
