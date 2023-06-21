using Random: Random
import Random.seed!
using MLJ
using MLJBase
using MLJFlux
using Flux
using StableRNGs

function basictest(ModelType, X, y, builder, optimiser, threshold, accel)
    ModelType_str = string(ModelType)
    ModelType_ex = Meta.parse(ModelType_str)
    accel_ex = Meta.parse(string(accel))
    optimiser = deepcopy(optimiser)

    eval(
        quote
            stable_rng = StableRNGs.StableRNG(123)

            model = $ModelType_ex(;
                builder=$builder,
                optimiser=$optimiser,
                acceleration=$accel_ex,
                rng=stable_rng,
            )

            fitresult, cache, _report = MLJBase.fit(model, 0, $X, $y)

            history = _report.training_losses
            @test length(history) == model.epochs + 1

            # test improvement in training loss:
            @test history[end] < $threshold * history[1]

            # increase iterations and check update is incremental:
            model.epochs = model.epochs + 3

            fitresult, cache, _report = @test_logs(
                (:info, r""), # one line of :info per extra epoch
                (:info, r""),
                (:info, r""),
                MLJBase.update(model, 2, fitresult, cache, $X, $y)
            )

            @test :chain in keys(MLJBase.fitted_params(model, fitresult))

            yhat = MLJBase.predict(model, fitresult, $X)

            history = _report.training_losses
            @test length(history) == model.epochs + 1

            # start fresh with small epochs:
            model = $ModelType_ex(;
                builder=$builder,
                optimiser=$optimiser,
                epochs=2,
                acceleration=$accel_ex,
                rng=stable_rng,
            )

            fitresult, cache, _report = MLJBase.fit(model, 0, $X, $y)

            # change batch_size and check it performs cold restart:
            model.batch_size = 2
            fitresult, cache, _report = @test_logs(
                (:info, r""), # one line of :info per extra epoch
                (:info, r""),
                MLJBase.update(model, 2, fitresult, cache, $X, $y)
            )

            # change learning rate and check it does *not* restart:
            model.optimiser.eta /= 2
            fitresult, cache, _report = @test_logs(
                MLJBase.update(model, 2, fitresult, cache, $X, $y)
            )

            # set `optimiser_changes_trigger_retraining = true` and change
            # learning rate and check it does restart:
            model.optimiser_changes_trigger_retraining = true
            model.optimiser.eta /= 2
            @test_logs(
                (:info, r""), # one line of :info per extra epoch
                (:info, r""),
                MLJBase.update(model, 2, fitresult, cache, $X, $y)
            )
        end,
    )

    return true
end

seed!(1234)
N = 300
X = MLJBase.table(rand(Float32, N, 4));
ycont = 2 * X.x1 - X.x3 + 0.1 * rand(N)
m, M = minimum(ycont), maximum(ycont)
_, a, b, _ = collect(range(m; stop=M, length=4))
y = categorical(
    map(ycont) do η
        if η < 0.9 * a
            'a'
        elseif η < 1.1 * b
            'b'
        else
            'c'
        end
    end,
);

builder = MLJFlux.MLP(; hidden=(16, 8), σ=Flux.relu)
optimizer = Flux.Optimise.Adam(0.03)

@test basictest(LaplaceApproximation, X, y, builder, optimizer, 0.9, CPU1())
