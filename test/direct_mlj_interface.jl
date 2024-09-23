using Random: Random
import Random.seed!
using MLJBase: MLJBase, categorical
using Flux
using StableRNGs


@testset "Regression" begin
    function basictest_regression(X, y, builder, optimiser, threshold)
        optimiser = deepcopy(optimiser)

        stable_rng = StableRNGs.StableRNG(123)

        model = LaplaceRegression(;
            builder=builder,
            optimiser=optimiser,
            acceleration=MLJBase.CPUThreads(),
            loss=Flux.Losses.mse,
            rng=stable_rng,
            lambda=-1.0,
            alpha=-1.0,
            epochs=-1,
            batch_size=-1,
            subset_of_weights=:incorrect,
            hessian_structure=:incorrect,
            backend=:incorrect,
            ret_distr=true,
        )

        fitresult, cache, _report = MLJBase.fit(model, 0, X, y)

        history = _report.training_losses
        @test length(history) == model.epochs + 1

        # increase iterations and check update is incremental:
        model.epochs = model.epochs + 3

        fitresult, cache, _report = @test_logs(
            (:info, r""), # one line of :info per extra epoch
            (:info, r""),
            (:info, r""),
            MLJBase.update(model, 2, fitresult, cache, X, y)
        )

        @test :chain in keys(MLJBase.fitted_params(model, fitresult))

        history = _report.training_losses
        @test length(history) == model.epochs + 1

        yhat = MLJBase.predict(model, fitresult, X)

        # start fresh with small epochs:
        model = LaplaceRegression(;
            builder=builder,
            optimiser=optimiser,
            epochs=2,
            acceleration=CPU1(),
            rng=stable_rng,
        )

        fitresult, cache, _report = MLJBase.fit(model, 0, X, y)

        # change batch_size and check it performs cold restart:
        model.batch_size = 2
        fitresult, cache, _report = @test_logs(
            (:info, r""), # one line of :info per extra epoch
            (:info, r""),
            MLJBase.update(model, 2, fitresult, cache, X, y)
        )

        # change learning rate and check it does *not* restart:
        model.optimiser.eta /= 2
        fitresult, cache, _report = @test_logs(
            MLJBase.update(model, 2, fitresult, cache, X, y)
        )

        # set `optimiser_changes_trigger_retraining = true` and change
        # learning rate and check it does restart:
        model.optimiser_changes_trigger_retraining = true
        model.optimiser.eta /= 2
        @test_logs(
            (:info, r""), # one line of :info per extra epoch
            (:info, r""),
            MLJBase.update(model, 2, fitresult, cache, X, y)
        )

        return true
    end

    seed!(1234)
    N = 300
    X = MLJBase.table(rand(Float32, N, 4))
    ycont = 2 * X.x1 - X.x3 + 0.1 * rand(N)
    builder = MLJFlux.MLP(; hidden=(16, 8), σ=Flux.relu)
    optimiser = Flux.Optimise.Adam(0.03)

    @test basictest_regression(X, ycont, builder, optimiser, 0.9)
end