function _branch_matches_record(branch::BranchProjectionResult, record::PruneRecord)
    for index in eachindex(record.assignments)
        record.assignments[index] == 0 && continue
        branch.choices[index] == record.assignments[index] || return false
    end
    return true
end

function _branch_could_beat_incumbent(
    branch::BranchProjectionResult,
    record::PruneRecord,
    atol::Float64,
)
    branch.accepted || return false
    branch.objective < record.incumbent_objective - atol && return true
    return abs(branch.objective - record.incumbent_objective) <= atol &&
           !isempty(record.incumbent_choices) &&
           isless(Tuple(branch.choices), Tuple(record.incumbent_choices))
end

"""
    audit_pruning(search, oracle; atol=1e-7)

Audit every lazy-search prune against stored exhaustive branch results. This is
an experimental validation tool, not part of the production search path.
"""
function audit_pruning(
    search::LazySearchResult,
    oracle::ExactProjectionResult;
    atol::Real = 1e-7,
)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative."))
    length(oracle.branch_results) == oracle.total_branches ||
        throw(ArgumentError("Oracle must store every branch result for pruning audit."))

    unsafe_prunes = 0
    unsafe_pruned_branches = 0
    for record in search.prune_records
        matching = filter(
            branch -> _branch_matches_record(branch, record),
            oracle.branch_results,
        )

        unsafe_count =
            if record.reason == :infeasible
                count(branch -> branch.accepted, matching)
            elseif record.reason == :bound
                count(
                    branch -> _branch_could_beat_incumbent(
                        branch,
                        record,
                        Float64(atol),
                    ),
                    matching,
                )
            else
                length(matching)
            end

        if unsafe_count > 0
            unsafe_prunes += 1
            unsafe_pruned_branches += unsafe_count
        end
    end

    status_match = search.status == oracle.status
    objective_match =
        search.status == :optimal && oracle.status == :optimal ?
        isapprox(search.objective, oracle.objective; atol = atol, rtol = atol) :
        search.status == oracle.status
    winner_feasible =
        search.status == :infeasible ||
        (search.report !== nothing && search.report.feasible)

    return PruningAudit(
        status_match,
        objective_match,
        winner_feasible,
        length(search.prune_records) == search.pruned_nodes,
        length(search.prune_records),
        unsafe_prunes,
        unsafe_pruned_branches,
    )
end

