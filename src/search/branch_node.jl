"""One partial assignment in the lazy best-first search."""
struct BranchSearchNode
    assignments::Vector{Int}
    depth::Int
    certified_lower_bound::Float64
    priority_value::Float64
    relaxation_y::Vector{Float64}
    serial::Int
end

"""Result of the continuous relaxation for one partial assignment."""
struct NodeRelaxationResult
    y::Vector{Float64}
    primal_objective::Float64
    certified_lower_bound::Float64
    solver_status::Any
    status::Symbol
    constraint_violation::Float64
    solve_time::Float64
    message::Union{Nothing, String}
end

"""Audit record for one safely removed subtree."""
struct PruneRecord
    assignments::Vector{Int}
    depth::Int
    reason::Symbol
    certified_lower_bound::Float64
    incumbent_objective::Float64
    incumbent_choices::Vector{Int}
    branch_count::Int
end

"""
    LazySearchResult

Result and diagnostics from certified lazy branch search.
"""
struct LazySearchResult
    y::Vector{Float64}
    choices::Vector{Int}
    objective::Float64
    distance::Float64
    status::Symbol
    report::Union{Nothing, FeasibilityReport}
    total_branches::Int
    leaf_solves::Int
    feasible_leaves::Int
    infeasible_leaves::Int
    unresolved_leaves::Int
    relaxation_solves::Int
    relaxation_failures::Int
    explored_nodes::Int
    created_nodes::Int
    pruned_nodes::Int
    pruned_branches::Int
    bound_prunes::Int
    infeasibility_prunes::Int
    max_open_nodes::Int
    rule_order::Vector{Int}
    termination_reason::Symbol
    solve_time::Float64
    prune_records::Vector{PruneRecord}
end

is_certified(result::LazySearchResult) = result.status in (:optimal, :infeasible)

function Base.show(io::IO, result::LazySearchResult)
    print(
        io,
        "LazySearchResult(status=$(result.status), objective=$(result.objective), " *
        "choices=$(result.choices), leaf_solves=$(result.leaf_solves)/$(result.total_branches), " *
        "pruned_branches=$(result.pruned_branches))",
    )
end

"""Comparison of lazy-search prune decisions against exhaustive ground truth."""
struct PruningAudit
    status_match::Bool
    objective_match::Bool
    winner_feasible::Bool
    all_prunes_recorded::Bool
    audited_prunes::Int
    unsafe_prunes::Int
    unsafe_pruned_branches::Int
end

audit_passed(audit::PruningAudit) =
    audit.status_match &&
    audit.objective_match &&
    audit.winner_feasible &&
    audit.all_prunes_recorded &&
    audit.unsafe_prunes == 0


mutable struct _SearchNodeHeap
    nodes::Vector{BranchSearchNode}
end

_SearchNodeHeap() = _SearchNodeHeap(BranchSearchNode[])
Base.isempty(heap::_SearchNodeHeap) = isempty(heap.nodes)
Base.length(heap::_SearchNodeHeap) = length(heap.nodes)

function _node_precedes(left::BranchSearchNode, right::BranchSearchNode)
    left.priority_value != right.priority_value &&
        return left.priority_value < right.priority_value
    left.certified_lower_bound != right.certified_lower_bound &&
        return left.certified_lower_bound < right.certified_lower_bound
    left.depth != right.depth && return left.depth > right.depth
    return left.serial < right.serial
end

function _heap_push!(heap::_SearchNodeHeap, node::BranchSearchNode)
    push!(heap.nodes, node)
    index = length(heap.nodes)
    while index > 1
        parent = index ÷ 2
        _node_precedes(heap.nodes[index], heap.nodes[parent]) || break
        heap.nodes[index], heap.nodes[parent] = heap.nodes[parent], heap.nodes[index]
        index = parent
    end
    return heap
end

function _heap_pop!(heap::_SearchNodeHeap)
    isempty(heap) && throw(ArgumentError("Cannot pop an empty search heap."))
    first_node = heap.nodes[1]
    last_node = pop!(heap.nodes)
    isempty(heap) && return first_node

    heap.nodes[1] = last_node
    index = 1
    while true
        left = 2 * index
        right = left + 1
        left > length(heap.nodes) && break
        child =
            right <= length(heap.nodes) &&
            _node_precedes(heap.nodes[right], heap.nodes[left]) ? right : left
        _node_precedes(heap.nodes[child], heap.nodes[index]) || break
        heap.nodes[index], heap.nodes[child] = heap.nodes[child], heap.nodes[index]
        index = child
    end
    return first_node
end

