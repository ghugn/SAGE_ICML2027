"""
    SwitchingCertificate

Numerical certificate for stability of the winning disjunct branch throughout
an Euclidean input ball. `:not_certified` means only that the sufficient test
was inconclusive; it is not evidence that a switch occurs.
"""
struct SwitchingCertificate
    center::Vector{Float64}
    radius::Float64
    status::Symbol
    winner::Vector{Int}
    winner_point::Vector{Float64}
    winner_upper_distance::Float64
    required_margin::Float64
    minimum_margin::Float64
    total_competitor_branches::Int
    stable_pruned_branches::Int
    competitive_branches::Int
    groups::Vector{CompetitorGroup}
    rule_order::Vector{Int}
    center_status::Symbol
    center_solve_time::Float64
    certificate_time::Float64
    message::Union{Nothing, String}
end

"""Return true only when branch stability over the full ball was certified."""
certifies_stability(certificate::SwitchingCertificate) = certificate.status == :stable
is_certified(certificate::SwitchingCertificate) = certifies_stability(certificate)

"""
    certified_local_radius(certificate; safety_factor=0.45, max_fraction=0.95)

Return a conservative local radius implied by the Phase 3 distance margin.
The strict certificate tolerance is removed before applying `safety_factor`;
the default `0.45 < 0.5` leaves a ten-percent buffer relative to the sufficient
condition `margin > 2radius + certificate_tol`. A zero result means that no
positive local ball is certified by the available bound.
"""
function certified_local_radius(
    certificate::SwitchingCertificate;
    safety_factor::Real = 0.45,
    max_fraction::Real = 0.95,
)
    isfinite(safety_factor) && 0.0 < safety_factor < 0.5 ||
        throw(ArgumentError("safety_factor must be finite and lie in (0, 0.5)."))
    isfinite(max_fraction) && 0.0 < max_fraction <= 1.0 ||
        throw(ArgumentError("max_fraction must be finite and lie in (0, 1]."))
    certificate.center_status == :optimal || return 0.0
    isfinite(certificate.minimum_margin) || return 0.0
    certificate_tolerance = max(
        certificate.required_margin - 2.0 * certificate.radius,
        0.0,
    )
    available_margin = max(certificate.minimum_margin - certificate_tolerance, 0.0)
    return min(
        Float64(safety_factor) * available_margin,
        Float64(max_fraction) * certificate.radius,
    )
end

function Base.show(io::IO, certificate::SwitchingCertificate)
    print(
        io,
        "SwitchingCertificate(status=$(certificate.status), radius=$(certificate.radius), " *
        "winner=$(certificate.winner), competitive_branches=$(certificate.competitive_branches)/" *
        "$(certificate.total_competitor_branches))",
    )
end

function _certificate_center_status(result)
    result.status == :infeasible && return :center_infeasible
    result.status == :optimal && return :optimal
    return :center_uncertified
end

function _empty_switching_certificate(
    yhat::Vector{Float64},
    radius::Float64,
    status::Symbol,
    center_status::Symbol,
    center_solve_time::Float64,
    certificate_time::Float64,
    message::String,
)
    return SwitchingCertificate(
        yhat, radius, status, Int[], Float64[], Inf, Inf, -Inf,
        0, 0, 0, CompetitorGroup[], Int[], center_status,
        center_solve_time, certificate_time, message,
    )
end

"""
    switching_certificate(model, yhat, radius; kwargs...)

Certify that the exact winning branch at `yhat` remains the winner for every
input in the Euclidean ball `norm(delta) <= radius`.

For every disjoint competitor group `G`, the sufficient condition is

    lower_distance(G) - upper_distance(winner) > 2radius + certificate_tol.

The lower distances are analytic geometric bounds and therefore never use a
solver primal objective as a pruning certificate. If the condition is not met,
the group remains in `competitor_set(result)`.
"""
function switching_certificate(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real},
    radius::Real;
    center_result = nothing,
    solver = nothing,
    rule_ordering = nothing,
    feasibility_tol::Real = 1e-6,
    certificate_tol::Real = 1e-9,
    tie_tol::Real = 1e-10,
    prune_tol::Real = 1e-10,
    max_nodes::Int = 1_000_000,
)
    length(yhat) == model.n_outputs ||
        throw(DimensionMismatch("yhat dimension does not match the model."))
    all(isfinite, yhat) || throw(ArgumentError("yhat must contain only finite values."))
    isfinite(radius) && radius >= 0 ||
        throw(ArgumentError("radius must be finite and nonnegative."))
    feasibility_tol >= 0 || throw(ArgumentError("feasibility_tol must be nonnegative."))
    certificate_tol >= 0 || throw(ArgumentError("certificate_tol must be nonnegative."))

    center = Float64.(collect(yhat))
    computed_center = center_result === nothing
    result = computed_center ? lazy_branch_search(
        model,
        center;
        solver = solver,
        rule_ordering = rule_ordering,
        feasibility_tol = feasibility_tol,
        tie_tol = tie_tol,
        prune_tol = prune_tol,
        max_nodes = max_nodes,
    ) : center_result
    certificate_started = time_ns()

    result isa Union{LazySearchResult, ExactProjectionResult} ||
        throw(ArgumentError(
            "center_result must be a LazySearchResult or ExactProjectionResult.",
        ))
    center_status = _certificate_center_status(result)
    center_solve_time = result isa LazySearchResult ? result.solve_time : 0.0

    if center_status != :optimal
        elapsed = (time_ns() - certificate_started) / 1e9
        message = center_status == :center_infeasible ?
            "The center problem is certified infeasible, so no winning branch exists." :
            "The center winner is not globally certified."
        return _empty_switching_certificate(
            center, Float64(radius), center_status, center_status,
            center_solve_time, elapsed, message,
        )
    end

    length(result.choices) == length(model.disjunctions) ||
        throw(ArgumentError("center_result choices do not match the model."))
    length(result.y) == model.n_outputs ||
        throw(ArgumentError("center_result point does not match the model."))
    result.report !== nothing && result.report.feasible ||
        throw(ArgumentError("center_result does not contain a feasible winner."))
    _selected_branch_violation(model, result.y, result.choices) <= feasibility_tol ||
        throw(ArgumentError("center_result point fails its selected branch constraints."))
    measured_objective = 0.5 * sum(abs2, result.y .- center)
    isapprox(
        result.objective,
        measured_objective;
        atol = max(Float64(certificate_tol), 1e-10),
        rtol = 1e-7,
    ) || throw(ArgumentError(
        "center_result objective is inconsistent with yhat; reuse only a result computed at this center.",
    ))

    order = if rule_ordering === nothing && result isa LazySearchResult
        copy(result.rule_order)
    else
        _resolve_search_order(model, rule_ordering)
    end
    structural_groups = competitor_partition(
        model,
        result.choices;
        rule_ordering = order,
    )

    # The accepted branch point is a constructive numerical upper bound. The
    # certificate tolerance is kept separate and applied to the distance gap.
    winner_upper = sqrt(sum(abs2, result.y .- center))
    required_margin = 2.0 * Float64(radius) + Float64(certificate_tol)
    groups = CompetitorGroup[]
    stable_pruned_branches = 0
    minimum_margin = Inf

    for structural in structural_groups
        lower_objective = geometric_lower_bound(model, center, structural.assignments)
        lower_distance = sqrt(2.0 * lower_objective)
        margin = lower_distance - winner_upper
        stable = margin > required_margin
        stable && (stable_pruned_branches += structural.branch_count)
        minimum_margin = min(minimum_margin, margin)
        push!(
            groups,
            CompetitorGroup(
                structural.assignments,
                structural.divergence_rule,
                structural.divergence_depth,
                structural.branch_count,
                lower_objective,
                lower_distance,
                margin,
                stable,
            ),
        )
    end

    total_competitors = num_branches(model) - 1
    competitive = total_competitors - stable_pruned_branches
    status = competitive == 0 ? :stable : :not_certified
    message = status == :stable ? nothing :
        "At least one competitor group lacks the strict distance margin; no switch is asserted."
    elapsed = (time_ns() - certificate_started) / 1e9

    return SwitchingCertificate(
        center,
        Float64(radius),
        status,
        copy(result.choices),
        copy(result.y),
        winner_upper,
        required_margin,
        minimum_margin,
        total_competitors,
        stable_pruned_branches,
        competitive,
        groups,
        order,
        center_status,
        center_solve_time,
        elapsed,
        message,
    )
end

