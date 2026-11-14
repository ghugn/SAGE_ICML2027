module DisjunctiveNet

include("core/types.jl")
include("core/utils.jl")
include("core/feasibility.jl")

include("modeling/modeling.jl")
include("modeling/jump_conversion.jl")
include("modeling/standard_form.jl")

include("formulations/convex_hull.jl")
include("formulations/cnf_hull.jl")
include("formulations/partial_dnf_hull.jl")

include("backend/projection_backend.jl")
include("backend/differentiation.jl")

include("exact/branch_types.jl")
include("exact/branch_projection.jl")
include("exact/exhaustive_oracle.jl")

include("search/branch_node.jl")
include("search/node_relaxation.jl")
include("search/lazy_branch_search.jl")
include("search/pruning_audit.jl")

include("certificates/competitor_set.jl")
include("certificates/branch_margin.jl")

include("gradients/smoothing_distribution.jl")
include("gradients/local_implicit.jl")
include("gradients/switching_correction.jl")
include("gradients/geometry_sampling.jl")
include("gradients/sage_estimator.jl")

include("layers/projection_layer.jl")
include("layers/flux.jl")
include("layers/constrained_model.jl")

include("display/display.jl")


export DisjunctiveModel
export DisjunctiveProjectionLayer
export ProjectionConfig
export FeasibilityReport

export LinearConstraint
export Disjunction
export StandardDisjunctiveModel
export ConvexHullForm
export ConvexHullScenario
export PartialDNFHullForm
export partial_dnf_hull_form

export CNFConvexHullForm
export CNFConvexHullBlock
export cnf_hull_form
export num_blocks

export output_dimension
export lower_bounds
export upper_bounds
export set_bounds!
export add_linear_constraint!
export add_disjunction!
export linear_constraints
export disjunctions
export output_variables
export linear_constraint_from_jump
export constraint_violation
export check_feasibility
export is_logically_feasible

export standard_form
export convex_hull_form
export num_scenarios
export scenario_has_interior_point

export projection_mode

export ProjectionResult
export BranchProjectionResult
export ExactProjectionResult
export project
export build_projection_model
export project_branch
export exhaustive_project
export exact_project
export num_branches
export is_certified
export BranchSearchNode
export NodeRelaxationResult
export PruneRecord
export LazySearchResult
export PruningAudit
export geometric_lower_bound
export solve_node_relaxation
export lazy_branch_search
export audit_pruning
export audit_passed
export CompetitorGroup
export SwitchingCertificate
export competitor_partition
export competitor_set
export switching_certificate
export certifies_stability
export certified_local_radius
export FixedBranchGradientResult
export RelevantProjectionResult
export SAGESampleRecord
export SAGEEstimate
export SAGEConfig
export GeometrySampleBatch
export CertifiedGeometrySampleBatch
export SwitchingGeometry
export sample_ball_smoothing
export sample_antithetic_ball_smoothing
export sample_geometry_smoothing
export sample_certified_geometry_smoothing
export ball_smoothing_score
export geometry_proposal_ratio
export scaled_ball_proposal_ratio
export certified_geometry_proposal_ratio
export switching_geometry
export fixed_branch_loss_gradient
export project_relevant_branches
export sage_gradient
export projection_formulation
export ConstrainedFluxModel
export constrained_model

export print_model
export print_hull
export print_projection_model

export count_constraints
export model_size
export product_disjunct_count
export formulation_summary
export benchmark_projection

end
