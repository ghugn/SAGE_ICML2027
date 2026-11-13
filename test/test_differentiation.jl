@testset "Differentiation" begin
        dm = DisjunctiveModel(2)

        set_bounds!(
            dm,
            lower = [0.0, 0.0],
            upper = [1.0, 1.0],
        )

        # Simple global constraint: y1 + y2 >= 0.8
        add_linear_constraint!(dm, [1.0, 1.0], :>=, 0.8)

        d1 = [
            LinearConstraint([1.0, 0.0], :<=, 0.25),
        ]

        d2 = [
            LinearConstraint([1.0, 0.0], :>=, 0.75),
        ]

        add_disjunction!(dm, d1, d2)

        layer = DisjunctiveProjectionLayer(dm)

        yhat = [0.5, 0.1]

        grad = Zygote.gradient(y -> sum(layer(y)), yhat)[1]

        @test grad !== nothing
        @test length(grad) == 2
        @test all(isfinite, grad)
    end

@testset "DiffOpt gradient matches central differences away from switches" begin
    dm = DisjunctiveModel(2)
    add_linear_constraint!(dm, [1.0, 1.0], :(==), 1.0)
    layer = DisjunctiveProjectionLayer(dm; formulation = :cnf)

    yhat = [0.7, 0.1]
    target = [0.2, 0.8]
    loss(z) = sum(abs2, layer(z) .- target)

    implicit_gradient = Zygote.gradient(loss, yhat)[1]
    step = 1e-5
    finite_difference = [
        begin
            direction = zeros(2)
            direction[j] = 1.0
            (loss(yhat .+ step .* direction) - loss(yhat .- step .* direction)) /
            (2 * step)
        end
        for j in eachindex(yhat)
    ]

    @test all(isfinite, implicit_gradient)
    @test isapprox(implicit_gradient, finite_difference; atol = 1e-5, rtol = 1e-5)
end
