using ChainRulesCore
using DiffOpt
using IterativeSolvers
using JuMP: ParameterRef
using LinearAlgebra
import MathOptInterface as MOI


"""
Linear solver used only when DiffOpt's direct KKT solve is singular.

LSQR returns the minimum-norm least-squares solution for rank-deficient systems,
which is the appropriate deterministic fallback at degenerate active sets.
"""
struct RobustKKTSolver
    atol::Float64
    btol::Float64
end

RobustKKTSolver() = RobustKKTSolver(1e-10, 1e-10)

function DiffOpt.QuadraticProgram.solve_system(
    solver::RobustKKTSolver,
    lhs,
    rhs,
    iterative::Bool,
)
    if !iterative
        try
            return lhs \ rhs
        catch err
            if !(err isa SingularException)
                rethrow()
            end
        end
    end

    return IterativeSolvers.lsqr(
        lhs,
        rhs;
        atol = solver.atol,
        btol = solver.btol,
        maxiter = max(100, 4 * maximum(size(lhs))),
    )
end


function _install_robust_kkt_solver!(opt_model::JuMP.Model)
    backend = JuMP.backend(opt_model)
    bridge_optimizer = backend.optimizer
    diff_optimizer = bridge_optimizer.model

    diff_optimizer isa DiffOpt.Optimizer ||
        error("Unexpected optimizer stack; cannot install the robust KKT solver.")
    diff_optimizer.diff === nothing &&
        error("DiffOpt did not construct a differentiation model.")

    MOI.set(
        diff_optimizer.diff,
        DiffOpt.QuadraticProgram.LinearAlgebraSolver(),
        RobustKKTSolver(),
    )
    return nothing
end


function _project_pullback(result::ProjectionResult, yhat, dresult)
    dy = _projection_result_y_tangent(dresult, result.y)

    if result.status != MOI.OPTIMAL
        return zeros(Float64, length(yhat))
    end

    opt_model = result.model
    y_var = opt_model[:y]
    yhat_param = opt_model[:yhat_param]

    MOI.set.(
        opt_model,
        DiffOpt.ReverseVariablePrimal(),
        y_var,
        Float64.(dy),
    )

    try
        DiffOpt.reverse_differentiate!(opt_model)
    catch err
        if err isa SingularException
            _install_robust_kkt_solver!(opt_model)
            DiffOpt.reverse_differentiate!(opt_model)
        else
            rethrow()
        end
    end

    yhat_refs = [ParameterRef(yhat_param[j]) for j in eachindex(yhat)]
    raw_grad = MOI.get.(opt_model, DiffOpt.ReverseConstraintSet(), yhat_refs)

    return Float64[g.value for g in raw_grad]
end


function ChainRulesCore.rrule(
    ::typeof(project),
    hull::ConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    result = project(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    function pullback(dresult)
        grad = _project_pullback(result, yhat, dresult)
        return NoTangent(), NoTangent(), grad
    end

    return result, pullback
end


function ChainRulesCore.rrule(
    ::typeof(project),
    hull::CNFConvexHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    result = project(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    function pullback(dresult)
        grad = _project_pullback(result, yhat, dresult)
        return NoTangent(), NoTangent(), grad
    end

    return result, pullback
end

function ChainRulesCore.rrule(
    ::typeof(project),
    hull::PartialDNFHullForm,
    yhat::AbstractVector{<:Real};
    solver = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    result = project(
        hull,
        yhat;
        solver = solver,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    function pullback(dresult)
        grad = _project_pullback(result, yhat, dresult)
        return NoTangent(), NoTangent(), grad
    end

    return result, pullback
end


function ChainRulesCore.rrule(
    ::typeof(project),
    model::DisjunctiveModel,
    yhat::AbstractVector{<:Real};
    formulation::Symbol = :dnf,
    solver = nothing,
    num_dnf_rules::Int = -1,
    rule_ordering = nothing,
    y_regularization::Real = 0.0,
    ycopy_regularization::Real = 0.0,
    gamma_regularization::Real = 0.0,
    anchor_regularization::Real = 1e-3,
)
    result = project(
        model,
        yhat;
        formulation = formulation,
        solver = solver,
        num_dnf_rules = num_dnf_rules,
        rule_ordering = rule_ordering,
        y_regularization = y_regularization,
        ycopy_regularization = ycopy_regularization,
        gamma_regularization = gamma_regularization,
        anchor_regularization = anchor_regularization,
    )

    function pullback(dresult)
        grad = _project_pullback(result, yhat, dresult)
        return NoTangent(), NoTangent(), grad
    end

    return result, pullback
end


function _projection_result_y_tangent(dresult, y_template)
    if dresult isa ChainRulesCore.AbstractZero
        return zeros(Float64, length(y_template))
    end

    unthunked = ChainRulesCore.unthunk(dresult)

    if unthunked isa ProjectionResult
        return Float64.(unthunked.y)
    end

    if unthunked isa NamedTuple && haskey(unthunked, :y)
        return Float64.(unthunked.y)
    end

    if unthunked isa Tuple
        return Float64.(unthunked[1])
    end

    return Float64.(unthunked)
end


function ChainRulesCore.rrule(
    layer::DisjunctiveProjectionLayer,
    yhat::AbstractVector{<:Real},
)
    y = layer(yhat)

    function pullback(dy)
        dy_vec = _projection_result_y_tangent(dy, y)

        if layer.config.gradient == :straight_through
            return NoTangent(), dy_vec
        end

        _, pb = ChainRulesCore.rrule(
            project,
            layer.model,
            yhat;
            formulation = layer.config.formulation,
            solver = layer.config.solver,
            num_dnf_rules = layer.config.num_dnf_rules,
            rule_ordering = layer.config.rule_ordering,
            y_regularization = layer.config.y_regularization,
            ycopy_regularization = layer.config.ycopy_regularization,
            gamma_regularization = layer.config.gamma_regularization,
            anchor_regularization = layer.config.anchor_regularization,
        )

        dproject = ProjectionResult(
            dy_vec,
            Float64[],
            MOI.OPTIMAL,
            nothing,
        )

        _, _, grad_yhat = pb(dproject)

        return NoTangent(), grad_yhat
    end

    return y, pullback
end
