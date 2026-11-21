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
    legacy_source = first(runner.PHASE18_LEGACY_CACHE_SOURCES)
    @test all(runner.cache_source_compatible, runner.PHASE18_LEGACY_CACHE_SOURCES)
    @test !runner.cache_source_compatible("unknown-source")
    @test_throws ErrorException runner.interval([1.0, 2.0])
    @test runner.interval(fill(-1.0, 5)).upper == -1.0

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
                    perturbed_searches = 4, branch_solves = 8))
            end
        end
        summary = runner.report(directory, rows)
        @test summary.status == "COMPLETE"
        @test summary.safety_gate
        @test summary.protocol == "fixed_1024_updates"
        @test length(summary.comparisons) == 5
        @test all(value.difference == -0.1 for value in values(summary.comparisons))
        @test summary.aggregates["sage_n4/fixed"].runs == 5
        @test isfile(joinpath(directory, "phase18_runs.csv"))
        @test isfile(joinpath(directory, "phase18_summary.json"))
        partial = runner.report(directory, rows[1:3])
        @test partial.status == "PARTIAL"
        @test isempty(partial.comparisons)
    end

    mktempdir() do directory
        spec = runner.EB.default_spec(:sage_n4)
        signature = (schema = runner.PHASE18_SCHEMA, source = legacy_source,
            seed = 11, spec = spec, steps = 1024, tag = "fixed", wall_seconds = Inf)
        row = (seed = 11, method = "sage_n4", tag = "fixed")
        cache = joinpath(directory, "completed_methods", "seed_11_sage_n4_fixed.jls")
        runner.save_atomic(cache, (signature = signature, row = row))
        @test runner.collect_confirmatory_rows(directory) == [row]
        runner.save_atomic(joinpath(directory, "completed_methods", "duplicate.jls"),
            (signature = signature, row = row))
        @test_throws ErrorException runner.collect_confirmatory_rows(directory)
    end

    expected_signature = (schema = runner.PHASE18_SCHEMA,
        source = runner.source_fingerprint(), seed = 1, tag = "pilot")
    legacy_signature = merge(expected_signature,
        (source = legacy_source, threads = 4, wall_seconds = Inf))
    @test runner.signature_compatible(legacy_signature, expected_signature)
    @test !runner.signature_compatible(
        merge(legacy_signature, (seed = 2,)), expected_signature)

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
end
