if !isdefined(@__MODULE__, :Phase10JobShop)
    include(joinpath(@__DIR__, "..", "scripts", "phase10_jobshop_common.jl"))
end
import .Phase10JobShop
using Serialization

@testset "Phase 17 sample frontier and model artifacts" begin
    config = Phase10JobShop.JobShopConfig(n_jobs = 3, n_machines = 3)
    problem = Phase10JobShop.prepare_problem(
        17; config = config, train_size = 2, iid_size = 2, ood_size = 2,
    )
    network = Phase10JobShop.pretrain_network(
        problem, 17; epochs = 1, batch_size = 2,
    )
    for samples in (2, 4, 8)
        result = Phase10JobShop.train_method(
            problem, network, :sage_multiscale, 17;
            epochs = 1, max_steps = 1, samples = samples,
            radius = 0.5, evaluation_interval = 1,
        )
        evaluation = Phase10JobShop.evaluate_network(
            result.network, problem.ood; config = config,
        )
        @test evaluation.csat == 1.0
        @test evaluation.unresolved == 0
        @test result.logical_failures == 0
        @test 0 < result.mean_effective_sample_size <= samples + 1e-10
    end

    result = Phase10JobShop.train_method(
        problem, network, :ste, 17;
        epochs = 1, max_steps = 1, samples = 2,
        evaluation_interval = 1,
    )
    mktempdir() do directory
        path = joinpath(directory, "phase17_model.jls")
        open(path, "w") do io
            serialize(io, (schema = 1, network = result.network))
        end
        payload = open(deserialize, path)
        @test payload.schema == 1
        for name in (:W1, :b1, :W2, :b2, :W3, :b3)
            @test getproperty(payload.network, name) ==
                getproperty(result.network, name)
        end
    end
end
