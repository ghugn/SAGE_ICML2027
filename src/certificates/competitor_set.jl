"""One disjoint subtree in the partition of all branches except the center winner."""
struct CompetitorGroup
    assignments::Vector{Int}
    divergence_rule::Int
    divergence_depth::Int
    branch_count::Int
    lower_objective::Float64
    lower_distance::Float64
    distance_margin::Float64
    stable_pruned::Bool
end

"""
    competitor_partition(model, winner; rule_ordering=nothing)

Partition every complete branch other than `winner` into disjoint prefix
subtrees. A zero assignment means that the corresponding rule remains free.
The returned groups contain only structural information; their distance fields
are initialized to conservative placeholders and are populated by
`switching_certificate`.
"""
function competitor_partition(
    model::DisjunctiveModel,
    winner::AbstractVector{<:Integer};
    rule_ordering = nothing,
)
    _validate_branch_choices(model, winner)
    order = _resolve_search_order(model, rule_ordering)
    winner_choices = Int.(collect(winner))
    groups = CompetitorGroup[]

    for (depth, rule_index) in enumerate(order)
        for alternative in eachindex(model.disjunctions[rule_index].disjuncts)
            alternative == winner_choices[rule_index] && continue
            assignments = zeros(Int, length(model.disjunctions))
            for prefix_depth in 1:(depth - 1)
                prefix_rule = order[prefix_depth]
                assignments[prefix_rule] = winner_choices[prefix_rule]
            end
            assignments[rule_index] = alternative
            push!(
                groups,
                CompetitorGroup(
                    assignments,
                    rule_index,
                    depth,
                    _subtree_branch_count(model, assignments),
                    0.0,
                    0.0,
                    -Inf,
                    false,
                ),
            )
        end
    end

    expected = num_branches(model) - 1
    covered = sum(group.branch_count for group in groups; init = 0)
    covered == expected || error(
        "Internal competitor partition error: covered $(covered), expected $(expected).",
    )
    return groups
end

"""Return the groups that cannot be safely removed at the requested radius."""
competitor_set(certificate) =
    [group for group in certificate.groups if !group.stable_pruned]

