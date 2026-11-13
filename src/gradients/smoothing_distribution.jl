using Random
using LinearAlgebra

function _rand_unit_gamma(rng::AbstractRNG, shape::Float64)
    shape > 0 || throw(ArgumentError("Gamma shape must be positive."))
    if shape < 1.0
        return _rand_unit_gamma(rng, shape + 1.0) * rand(rng)^(1.0 / shape)
    end

    d = shape - 1.0 / 3.0
    c = inv(sqrt(9.0 * d))
    while true
        x = randn(rng)
        base = 1.0 + c * x
        base > 0.0 || continue
        v = base^3
        u = rand(rng)
        if u < 1.0 - 0.0331 * x^4 ||
           log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
            return d * v
        end
    end
end

"""
    sample_ball_smoothing(rng, dimension; beta=2.0)

Sample from the smooth compact-support density
`q(u) proportional to (1 - norm(u)^2)^beta` on the open Euclidean unit ball.
For this radial density, `norm(u)^2` follows `Beta(d/2, beta+1)`.
"""
function sample_ball_smoothing(
    rng::AbstractRNG,
    dimension::Integer;
    beta::Real = 2.0,
)
    d = Int(dimension)
    d > 0 || throw(ArgumentError("dimension must be positive."))
    isfinite(beta) && beta >= 2.0 ||
        throw(ArgumentError("beta must be finite and at least 2 for C1 compact support."))

    direction = randn(rng, d)
    direction_norm = norm(direction)
    while direction_norm == 0.0
        direction .= randn(rng, d)
        direction_norm = norm(direction)
    end
    direction ./= direction_norm

    radial_gamma = _rand_unit_gamma(rng, d / 2.0)
    boundary_gamma = _rand_unit_gamma(rng, Float64(beta) + 1.0)
    radius = sqrt(radial_gamma / (radial_gamma + boundary_gamma))
    return radius .* direction
end

sample_ball_smoothing(dimension::Integer; beta::Real = 2.0) =
    sample_ball_smoothing(Random.default_rng(), dimension; beta = beta)

"""Return `s_q(u) = -grad log q(u)` for the polynomial ball density."""
function ball_smoothing_score(u::AbstractVector{<:Real}; beta::Real = 2.0)
    isfinite(beta) && beta >= 2.0 ||
        throw(ArgumentError("beta must be finite and at least 2."))
    point = Float64.(collect(u))
    all(isfinite, point) || throw(ArgumentError("u must contain only finite values."))
    radius_squared = sum(abs2, point)
    radius_squared < 1.0 ||
        throw(ArgumentError("The smoothing score is defined only inside the unit ball."))
    return (2.0 * Float64(beta) / (1.0 - radius_squared)) .* point
end

