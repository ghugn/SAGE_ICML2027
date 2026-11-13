using DisjunctiveNet
using Printf

function separated_model(dimension)
    model = DisjunctiveModel(dimension)
    set_bounds!(model, lower = zeros(dimension), upper = ones(dimension))
    for coordinate in 1:dimension
        direction = zeros(dimension)
        direction[coordinate] = 1.0
        add_disjunction!(
            model,
            [LinearConstraint(direction, :<=, 0.2)],
            [LinearConstraint(direction, :>=, 0.8)],
        )
    end
    return model
end

# Warm Julia/JuMP/HiGHS and certificate compilation before measuring overhead.
warmup = separated_model(1)
warm_center = lazy_branch_search(warmup, [0.9])
switching_certificate(warmup, [0.9], 0.02; center_result = warm_center)

benchmark_model = separated_model(8)
benchmark_center = fill(0.9, 8)
center_result = lazy_branch_search(benchmark_model, benchmark_center)
switching_certificate(
    benchmark_model,
    benchmark_center,
    0.02;
    center_result = center_result,
)
certificate_ref = Ref{SwitchingCertificate}()
certificate_times = Float64[]
for _ in 1:21
    elapsed = @elapsed certificate_ref[] = switching_certificate(
        benchmark_model,
        benchmark_center,
        0.02;
        center_result = center_result,
    )
    push!(certificate_times, elapsed)
end
sort!(certificate_times)
certificate_elapsed = certificate_times[11]
certificate = certificate_ref[]

# Independent exhaustive grid audit on a tractable 2D instance.
audit_model = separated_model(2)
audit_center = [0.9, 0.9]
audit_radius = 0.05
audit_certificate = switching_certificate(audit_model, audit_center, audit_radius)
tested_points = Ref(0)
switches = Ref(0)
for dx in range(-audit_radius, audit_radius; length = 17)
    for dy in range(-audit_radius, audit_radius; length = 17)
        dx^2 + dy^2 <= audit_radius^2 + 1e-12 || continue
        perturbed = exhaustive_project(audit_model, audit_center .+ [dx, dy])
        tested_points[] += 1
        perturbed.choices == audit_certificate.winner || (switches[] += 1)
    end
end

# A symmetric center must remain explicitly inconclusive.
near_tie = switching_certificate(audit_model, [0.5, 0.5], 0.01)

@assert center_result.status == :optimal
@assert certificate.status == :stable
@assert certificate.competitive_branches == 0
@assert audit_certificate.status == :stable
@assert switches[] == 0
@assert near_tie.status == :not_certified
@assert near_tie.competitive_branches > 0

println("DisjunctiveNet Phase 3 switching-certificate verification")
println("Julia version: ", VERSION)
println("Benchmark dimension/branches: 8/", num_branches(benchmark_model))
println("Center search time: ", center_result.solve_time, " seconds")
println("Certificate-only median time (21 runs): ", certificate_elapsed, " seconds")
println("Certificate internal time: ", certificate.certificate_time, " seconds")
println("Stable-pruned competitor branches: ", certificate.stable_pruned_branches)
println("Competitive branches retained: ", certificate.competitive_branches)
println("Minimum distance margin: ", certificate.minimum_margin)
println("Required margin: ", certificate.required_margin)
println("2D exhaustive grid points: ", tested_points[])
println("Switches after stable claim: ", switches[])
println("Near-tie status: ", near_tie.status)
println("Near-tie competitive branches: ", near_tie.competitive_branches)
println("SWITCHING_CERTIFICATE_VERIFICATION=PASS")

