using DisjunctiveNet

function separated_model(dimension)
    dm = DisjunctiveModel(dimension)
    set_bounds!(dm, lower = zeros(dimension), upper = ones(dimension))
    for coordinate in 1:dimension
        direction = zeros(dimension)
        direction[coordinate] = 1.0
        add_disjunction!(
            dm,
            [LinearConstraint(direction, :<=, 0.2)],
            [LinearConstraint(direction, :>=, 0.8)],
        )
    end
    return dm
end

# Warm up Julia, JuMP, and HiGHS before reporting wall times.
warmup = separated_model(2)
lazy_branch_search(warmup, [0.9, 0.9])
exhaustive_project(warmup, [0.9, 0.9])

dimension = 8
dm = separated_model(dimension)
yhat = fill(0.9, dimension)

lazy_ref = Ref{LazySearchResult}()
oracle_ref = Ref{ExactProjectionResult}()
lazy_time = @elapsed lazy_ref[] = lazy_branch_search(dm, yhat)
oracle_time = @elapsed oracle_ref[] = exhaustive_project(dm, yhat)

lazy = lazy_ref[]
oracle = oracle_ref[]
audit = audit_pruning(lazy, oracle)
prune_fraction = lazy.pruned_branches / lazy.total_branches

println("DisjunctiveNet Phase 2 lazy-search verification")
println("Julia version: ", VERSION)
println("Dimension/rules: ", dimension)
println("Total branches: ", lazy.total_branches)
println("Lazy status: ", lazy.status)
println("Winning choices: ", lazy.choices)
println("Objective: ", lazy.objective)
println("Leaf solves: ", lazy.leaf_solves)
println("Relaxation solves: ", lazy.relaxation_solves)
println("Pruned branches: ", lazy.pruned_branches)
println("Prune fraction: ", prune_fraction)
println("Explored nodes: ", lazy.explored_nodes)
println("Unsafe prunes: ", audit.unsafe_prunes)
println("Unsafe pruned branches: ", audit.unsafe_pruned_branches)
println("All prunes recorded: ", audit.all_prunes_recorded)
println("Lazy time after warmup: ", lazy_time, " seconds")
println("Exhaustive time after warmup: ", oracle_time, " seconds")
println("Observed speed ratio exhaustive/lazy: ", oracle_time / lazy_time)

@assert lazy.status == :optimal
@assert oracle.status == :optimal
@assert isapprox(lazy.objective, oracle.objective; atol = 1e-8)
@assert lazy.report.feasible
@assert lazy.leaf_solves + lazy.pruned_branches == lazy.total_branches
@assert prune_fraction > 0.99
@assert audit_passed(audit)

println("LAZY_SEARCH_VERIFICATION=PASS")

