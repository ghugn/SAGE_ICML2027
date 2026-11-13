"""
    FeasibilityReport

Certificate for a point checked against the *original* disjunctive model.

`disjunction_violations[r]` is the smallest maximum constraint violation among
the disjuncts of rule `r`. `selected_disjuncts[r]` is the disjunct attaining
that value. The point is logically feasible only when every reported violation
is at most the requested tolerance.
"""
struct FeasibilityReport
    feasible::Bool
    tolerance::Float64
    max_violation::Float64
    bound_violation::Float64
    global_violation::Float64
    disjunction_violations::Vector{Float64}
    selected_disjuncts::Vector{Int}
end

function Base.show(io::IO, report::FeasibilityReport)
    print(
        io,
        "FeasibilityReport(feasible=$(report.feasible), " *
        "max_violation=$(report.max_violation), tolerance=$(report.tolerance))",
    )
end

"""
    constraint_violation(constraint, y)

Return the nonnegative residual by which `y` violates a linear constraint.
"""
function constraint_violation(
    constraint::LinearConstraint,
    y::AbstractVector{<:Real},
)
    length(y) == length(constraint.a) ||
        throw(DimensionMismatch("Point and constraint dimensions do not match."))

    value = sum(constraint.a[j] * y[j] for j in eachindex(y))
    if constraint.sense == :<=
        return max(Float64(value - constraint.b), 0.0)
    elseif constraint.sense == :>=
        return max(Float64(constraint.b - value), 0.0)
    else
        return Float64(abs(value - constraint.b))
    end
end

"""
    check_feasibility(model, y; atol = 1e-6)

Check `y` against bounds, global constraints, and the original Boolean logic of
`model`. Every disjunction is an OR over its disjuncts, while all disjunctions
must hold simultaneously.

This is intentionally different from checking feasibility in a CNF, partial-DNF,
or DNF convex-hull relaxation. An optimal relaxation solution can still have
`report.feasible == false`.
"""
function check_feasibility(
    model::DisjunctiveModel,
    y::AbstractVector{<:Real};
    atol::Real = 1e-6,
)
    length(y) == model.n_outputs ||
        throw(DimensionMismatch(
            "Expected a point of length $(model.n_outputs), got $(length(y)).",
        ))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative."))

    point = Float64.(collect(y))
    tolerance = Float64(atol)

    if !all(isfinite, point)
        n_rules = length(model.disjunctions)
        return FeasibilityReport(
            false,
            tolerance,
            Inf,
            Inf,
            Inf,
            fill(Inf, n_rules),
            fill(0, n_rules),
        )
    end

    bound_violation = 0.0
    for j in eachindex(point)
        if isfinite(model.lb[j])
            bound_violation = max(bound_violation, model.lb[j] - point[j])
        end
        if isfinite(model.ub[j])
            bound_violation = max(bound_violation, point[j] - model.ub[j])
        end
    end
    bound_violation = max(bound_violation, 0.0)

    global_violation = 0.0
    for constraint in model.constraints
        global_violation = max(
            global_violation,
            constraint_violation(constraint, point),
        )
    end

    rule_violations = Float64[]
    selected_disjuncts = Int[]
    for rule in model.disjunctions
        best_violation = Inf
        best_disjunct = 0
        for (index, disjunct) in enumerate(rule.disjuncts)
            violation = 0.0
            for constraint in disjunct
                violation = max(
                    violation,
                    constraint_violation(constraint, point),
                )
            end
            if violation < best_violation
                best_violation = violation
                best_disjunct = index
            end
        end
        push!(rule_violations, best_violation)
        push!(selected_disjuncts, best_disjunct)
    end

    max_violation = max(
        bound_violation,
        global_violation,
        isempty(rule_violations) ? 0.0 : maximum(rule_violations),
    )

    return FeasibilityReport(
        max_violation <= tolerance,
        tolerance,
        max_violation,
        bound_violation,
        global_violation,
        rule_violations,
        selected_disjuncts,
    )
end

"""
    is_logically_feasible(model, y; atol = 1e-6)

Return whether `y` satisfies the original disjunctive model.
"""
is_logically_feasible(model::DisjunctiveModel, y; atol::Real = 1e-6) =
    check_feasibility(model, y; atol = atol).feasible

