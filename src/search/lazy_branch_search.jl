function _resolve_search_order(model::DisjunctiveModel, ordering)
    n_rules = length(model.disjunctions)
    ordering === nothing && return collect(1:n_rules)

    if all(item -> item isa Integer, ordering)
        indices = Int.(collect(ordering))
        all(index -> 1 <= index <= n_rules, indices) ||
            throw(ArgumentError("Rule ordering contains an out-of-range index."))
        length(unique(indices)) == length(indices) ||
            throw(ArgumentError("Rule ordering contains duplicate indices."))
        append!(indices, [index for index in 1:n_rules if !(index in indices)])
        return indices
    end

    return _resolve_rule_order(standard_form(model), ordering)
end

function _subtree_branch_count(model::DisjunctiveModel, assignments::Vector{Int})
    count = 1
    for (rule_index, choice) in enumerate(assignments)
        if choice == 0
            count = Base.checked_mul(count, length(model.disjunctions[rule_index].disjuncts))
        end
    end
    return count
end

function _assignment_contains(assignments::Vector{Int}, choices::Tuple)
    for index in eachindex(assignments)
        assignments[index] == 0 && continue
        assignments[index] == choices[index] || return false
    end
    return true
end

function _unaccounted_subtree_count(
    model::DisjunctiveModel,
    assignments::Vector{Int},
    solved_choices::Set{Tuple},
)
    count = _subtree_branch_count(model, assignments)
    for choices in solved_choices
        _assignment_contains(assignments, choices) && (count -= 1)
    end
    return max(count, 0)
end

function _greedy_complete_choices(model::DisjunctiveModel, point::Vector{Float64})
    choices = Int[]
    for rule in model.disjunctions
        best_choice = 1
        best_violation = Inf
        for (choice, disjunct) in enumerate(rule.disjuncts)
            violation = maximum(
                (constraint_violation(constraint, point) for constraint in disjunct);
                init = 0.0,
            )
            if violation < best_violation
                best_violation = violation
                best_choice = choice
            end
        end
        push!(choices, best_choice)
    end
    return choices
end

function _lexicographically_better(
    candidate::BranchProjectionResult,
    incumbent::Union{Nothing, BranchProjectionResult},
    tie_tol::Float64,
)
    incumbent === nothing && return true
    candidate.objective < incumbent.objective - tie_tol && return true
    return abs(candidate.objective - incumbent.objective) <= tie_tol &&
           isless(Tuple(candidate.choices), Tuple(incumbent.choices))
end

"""
    lazy_branch_search(model, yhat; kwargs...)

Best-first search over partial disjunct assignments. Only analytic geometric
lower bounds and solver infeasibility statuses may remove a subtree. Partial-QP
primal objectives affect search order but never certify a bound prune. An
optional accepted `incumbent_result` at the same `yhat` is revalidated and can
seed the exact search without another branch projection.
"""
function lazy_branch_search(
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    rule_ordering = nothing,
    feasibility_tol::Real = 1e-6,
    tie_tol::Real = 1e-10,
    prune_tol::Real = 1e-10,
    max_nodes::Int = 1_000_000,
    seed_incumbent::Bool = true,
    incumbent_result = nothing,
    store_prune_records::Bool = true,
)
    length(yhat) == model.n_outputs ||
        throw(DimensionMismatch("yhat dimension does not match the model."))
    all(isfinite, yhat) || throw(ArgumentError("yhat must contain only finite values."))
    feasibility_tol >= 0 || throw(ArgumentError("feasibility_tol must be nonnegative."))
    tie_tol >= 0 || throw(ArgumentError("tie_tol must be nonnegative."))
    prune_tol >= 0 || throw(ArgumentError("prune_tol must be nonnegative."))
    max_nodes > 0 || throw(ArgumentError("max_nodes must be positive."))

    started = time_ns()
    order = _resolve_search_order(model, rule_ordering)
    total = num_branches(model)
    n_rules = length(model.disjunctions)

    leaf_solves = 0
    feasible_leaves = 0
    infeasible_leaves = 0
    unresolved_leaves = 0
    relaxation_solves = 0
    relaxation_failures = 0
    explored_nodes = 0
    created_nodes = 0
    pruned_nodes = 0
    pruned_branches = 0
    bound_prunes = 0
    infeasibility_prunes = 0
    max_open_nodes = 0
    serial = 0
    termination_reason = :complete
    prune_records = PruneRecord[]
    solved_choices = Set{Tuple}()
    incumbent::Union{Nothing, BranchProjectionResult} = nothing

    function consider_leaf!(assignments::Vector{Int})
        key = Tuple(assignments)
        key in solved_choices && return nothing
        branch = project_branch(
            model,
            yhat,
            assignments;
            solver = solver,
            feasibility_tol = feasibility_tol,
        )
        push!(solved_choices, key)
        leaf_solves += 1
        if branch.accepted
            feasible_leaves += 1
            _lexicographically_better(branch, incumbent, Float64(tie_tol)) &&
                (incumbent = branch)
        elseif branch.proven_infeasible
            infeasible_leaves += 1
        else
            unresolved_leaves += 1
        end
        return nothing
    end

    function record_prune!(
        assignments::Vector{Int},
        depth::Int,
        reason::Symbol,
        lower_bound::Float64,
    )
        count = _unaccounted_subtree_count(model, assignments, solved_choices)
        pruned_nodes += 1
        pruned_branches += count
        reason == :bound ? (bound_prunes += 1) : (infeasibility_prunes += 1)
        if store_prune_records
            push!(
                prune_records,
                PruneRecord(
                    copy(assignments),
                    depth,
                    reason,
                    lower_bound,
                    incumbent === nothing ? Inf : incumbent.objective,
                    incumbent === nothing ? Int[] : copy(incumbent.choices),
                    count,
                ),
            )
        end
        return nothing
    end

    function should_bound_prune(lower_bound::Float64)
        return incumbent !== nothing &&
               lower_bound > incumbent.objective + Float64(prune_tol)
    end

    if incumbent_result !== nothing
        incumbent_result isa BranchProjectionResult || throw(ArgumentError(
            "incumbent_result must be a BranchProjectionResult.",
        ))
        incumbent_result.accepted || throw(ArgumentError(
            "incumbent_result must be an accepted feasible branch projection.",
        ))
        _validate_branch_choices(model, incumbent_result.choices)
        length(incumbent_result.y) == model.n_outputs || throw(ArgumentError(
            "incumbent_result output dimension does not match the model.",
        ))
        all(isfinite, incumbent_result.y) || throw(ArgumentError(
            "incumbent_result output must be finite.",
        ))
        report = check_feasibility(
            model, incumbent_result.y; atol = feasibility_tol,
        )
        branch_violation = _selected_branch_violation(
            model, incumbent_result.y, incumbent_result.choices,
        )
        report.feasible && branch_violation <= feasibility_tol || throw(ArgumentError(
            "incumbent_result is not feasible for its selected model branch.",
        ))
        measured_objective = 0.5 * sum(abs2, incumbent_result.y .- Float64.(yhat))
        isapprox(
            incumbent_result.objective,
            measured_objective;
            atol = max(Float64(prune_tol), 1e-10),
            rtol = 1e-7,
        ) || throw(ArgumentError(
            "incumbent_result was not computed at the supplied search point.",
        ))
        incumbent = incumbent_result
        push!(solved_choices, Tuple(incumbent_result.choices))
        leaf_solves = 1
        feasible_leaves = 1
    end

    if n_rules == 0
        consider_leaf!(Int[])
        status =
            unresolved_leaves > 0 ? :uncertified :
            incumbent === nothing ? :infeasible : :optimal
        elapsed = (time_ns() - started) / 1e9
        return LazySearchResult(
            incumbent === nothing ? Float64[] : copy(incumbent.y),
            incumbent === nothing ? Int[] : copy(incumbent.choices),
            incumbent === nothing ? Inf : incumbent.objective,
            incumbent === nothing ? Inf : incumbent.distance,
            status,
            incumbent === nothing ? nothing : incumbent.report,
            total, leaf_solves, feasible_leaves, infeasible_leaves,
            unresolved_leaves, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            order, :complete, elapsed, prune_records,
        )
    end

    root_assignments = zeros(Int, n_rules)
    root_relaxation = solve_node_relaxation(
        model,
        yhat,
        root_assignments;
        solver = solver,
        feasibility_tol = feasibility_tol,
    )
    relaxation_solves += 1
    created_nodes += 1

    if root_relaxation.status == :infeasible
        record_prune!(root_assignments, 0, :infeasible, Inf)
        elapsed = (time_ns() - started) / 1e9
        return LazySearchResult(
            Float64[], Int[], Inf, Inf, :infeasible, nothing,
            total, 0, 0, 0, 0, relaxation_solves, 0, 0, created_nodes,
            pruned_nodes, pruned_branches, bound_prunes, infeasibility_prunes,
            0, order, :complete, elapsed, prune_records,
        )
    elseif root_relaxation.status == :unresolved
        relaxation_failures += 1
    elseif seed_incumbent && incumbent === nothing
        consider_leaf!(_greedy_complete_choices(model, root_relaxation.y))
    end

    heap = _SearchNodeHeap()
    serial += 1
    root_priority =
        root_relaxation.status == :optimal ? root_relaxation.primal_objective : -Inf
    _heap_push!(
        heap,
        BranchSearchNode(
            root_assignments,
            0,
            root_relaxation.certified_lower_bound,
            root_priority,
            root_relaxation.y,
            serial,
        ),
    )
    max_open_nodes = 1

    while !isempty(heap)
        if explored_nodes >= max_nodes
            termination_reason = :node_limit
            break
        end

        node = _heap_pop!(heap)
        explored_nodes += 1

        if should_bound_prune(node.certified_lower_bound)
            record_prune!(
                node.assignments,
                node.depth,
                :bound,
                node.certified_lower_bound,
            )
            continue
        end

        next_depth = node.depth + 1
        rule_index = order[next_depth]
        n_disjuncts = length(model.disjunctions[rule_index].disjuncts)

        for choice in 1:n_disjuncts
            child_assignments = copy(node.assignments)
            child_assignments[rule_index] = choice
            created_nodes += 1
            child_bound = geometric_lower_bound(model, yhat, child_assignments)

            if should_bound_prune(child_bound)
                record_prune!(child_assignments, next_depth, :bound, child_bound)
                continue
            end

            if next_depth == n_rules
                consider_leaf!(child_assignments)
                continue
            end

            relaxation = solve_node_relaxation(
                model,
                yhat,
                child_assignments;
                solver = solver,
                feasibility_tol = feasibility_tol,
            )
            relaxation_solves += 1

            if relaxation.status == :infeasible
                record_prune!(child_assignments, next_depth, :infeasible, Inf)
                continue
            elseif relaxation.status == :unresolved
                relaxation_failures += 1
            end

            if should_bound_prune(relaxation.certified_lower_bound)
                record_prune!(
                    child_assignments,
                    next_depth,
                    :bound,
                    relaxation.certified_lower_bound,
                )
                continue
            end

            serial += 1
            priority =
                relaxation.status == :optimal ? relaxation.primal_objective : -Inf
            _heap_push!(
                heap,
                BranchSearchNode(
                    child_assignments,
                    next_depth,
                    relaxation.certified_lower_bound,
                    priority,
                    relaxation.y,
                    serial,
                ),
            )
            max_open_nodes = max(max_open_nodes, length(heap))
        end
    end

    complete = isempty(heap) && termination_reason == :complete
    accounted = leaf_solves + pruned_branches
    if complete && accounted != total
        termination_reason = :accounting_error
        complete = false
    end

    status =
        !complete || unresolved_leaves > 0 ? :uncertified :
        incumbent === nothing ? :infeasible : :optimal
    elapsed = (time_ns() - started) / 1e9

    return LazySearchResult(
        incumbent === nothing ? Float64[] : copy(incumbent.y),
        incumbent === nothing ? Int[] : copy(incumbent.choices),
        incumbent === nothing ? Inf : incumbent.objective,
        incumbent === nothing ? Inf : incumbent.distance,
        status,
        incumbent === nothing ? nothing : incumbent.report,
        total,
        leaf_solves,
        feasible_leaves,
        infeasible_leaves,
        unresolved_leaves,
        relaxation_solves,
        relaxation_failures,
        explored_nodes,
        created_nodes,
        pruned_nodes,
        pruned_branches,
        bound_prunes,
        infeasibility_prunes,
        max_open_nodes,
        order,
        termination_reason,
        elapsed,
        prune_records,
    )
end

