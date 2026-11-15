using DisjunctiveNet
using Random
using Statistics

include(joinpath(@__DIR__, "phase10_jobshop_common.jl"))
using .Phase10JobShop

function deterministic_jobshop_fixture()
    config = JobShopConfig(n_jobs = 3, n_machines = 3)
    raw_durations = [3, 2, 4, 2, 5, 3, 4, 2, 3]
    machines = [1, 2, 3, 2, 3, 1, 3, 1, 2]
    horizon = sum(raw_durations)
    durations = Float64.(raw_durations) ./ horizon
    model = build_jobshop_model(durations, machines, config)
    center = Phase10JobShop._serial_dispatch(raw_durations, config) ./ horizon
    check_feasibility(model, center).feasible || error(
        "Deterministic parallel-verification center is not feasible.",
    )
    target = zeros(length(center))
    loss(y) = sum(abs2, y .- target)
    loss_gradient(y) = 2.0 .* (y .- target)
    return model, center, loss, loss_gradient
end

function timed_estimate(model, center, loss, loss_gradient, parallel_samples, stream)
    samples = sample_antithetic_ball_smoothing(
        MersenneTwister(stream), length(center), 32,
    )
    result = nothing
    elapsed = @elapsed result = sage_gradient(
        model,
        center,
        0.5,
        loss,
        loss_gradient;
        samples = samples,
        parallel_samples = parallel_samples,
        store_records = true,
    )
    return result, elapsed
end

function assert_parity(serial, threaded)
    serial.status == threaded.status == :ok || error(
        "Serial or threaded SAGE estimate was not resolved.",
    )
    serial.gradient == threaded.gradient || error(
        "Threaded SAGE gradient differs from the deterministic serial reduction.",
    )
    for field in (
        :reference_branch_solves,
        :perturbed_global_searches,
        :competitor_branch_solves,
        :sample_stability_certificates,
        :restricted_global_searches,
        :lazy_global_searches,
        :lazy_leaf_solves,
        :lazy_relaxation_solves,
        :switching_samples,
        :unresolved_samples,
    )
        getproperty(serial, field) == getproperty(threaded, field) || error(
            "Threaded SAGE accounting differs for $(field).",
        )
    end
    [record.u for record in serial.records] ==
        [record.u for record in threaded.records] || error(
            "Threaded SAGE record ordering differs from serial.",
        )
    return nothing
end

function main()
    Threads.nthreads() >= 2 || error(
        "Parallel verification requires JULIA_NUM_THREADS >= 2.",
    )
    model, center, loss, loss_gradient = deterministic_jobshop_fixture()

    # Compile both paths before timing.
    warm_serial, _ = timed_estimate(
        model, center, loss, loss_gradient, false, 17_001,
    )
    warm_threaded, _ = timed_estimate(
        model, center, loss, loss_gradient, true, 17_001,
    )
    assert_parity(warm_serial, warm_threaded)

    serial_times = Float64[]
    threaded_times = Float64[]
    observed_threads = 0
    for repetition in 1:3
        stream = 18_000 + repetition
        if isodd(repetition)
            serial, serial_seconds = timed_estimate(
                model, center, loss, loss_gradient, false, stream,
            )
            threaded, threaded_seconds = timed_estimate(
                model, center, loss, loss_gradient, true, stream,
            )
        else
            threaded, threaded_seconds = timed_estimate(
                model, center, loss, loss_gradient, true, stream,
            )
            serial, serial_seconds = timed_estimate(
                model, center, loss, loss_gradient, false, stream,
            )
        end
        assert_parity(serial, threaded)
        observed_threads = max(observed_threads, threaded.sample_threads_used)
        push!(serial_times, serial_seconds)
        push!(threaded_times, threaded_seconds)
    end

    serial_median = median(serial_times)
    threaded_median = median(threaded_times)
    speedup = serial_median / threaded_median
    parity_gate = true
    utilization_gate = observed_threads >= 2
    speed_gate = threaded_median < serial_median

    println("SAGE parallel-sample verification")
    println("Julia version/threads: ", VERSION, "/", Threads.nthreads())
    println("serial seconds: ", serial_times)
    println("threaded seconds: ", threaded_times)
    println("median speedup: ", speedup)
    println("sample threads used: ", observed_threads)
    println("parity gate: ", parity_gate ? "PASS" : "FAIL")
    println("utilization gate: ", utilization_gate ? "PASS" : "FAIL")
    println("speed gate: ", speed_gate ? "PASS" : "FAIL")
    passed = parity_gate && utilization_gate && speed_gate
    println("SAGE_PARALLEL_VERIFICATION=", passed ? "PASS" : "FAIL")
    passed || error("SAGE parallel-sample verification failed.")
end

main()
