using Random: Random
import Random.seed!
using MLJBase: MLJBase, categorical
import StatisticalMeasures as SM
using Flux
using StableRNGs
import LaplaceRedux: LaplaceClassifier, LaplaceRegressor
import MLJTestInterface

cv = MLJBase.CV(; nfolds=3)

@testset "Regression" begin
    @info " testing  interface for LaplaceRegressor"
    flux_model = Chain(Dense(4, 10, relu), Dense(10, 10, relu), Dense(10, 1))
    model = LaplaceRegressor(; model=flux_model, epochs=20)

    # Aliases:
    model.σ = model.observational_noise
    model.μ₀ = model.prior_mean
    model.P₀ = model.prior_precision_matrix
    @test model.observational_noise == model.σ
    @test model.prior_mean == model.μ₀
    @test model.prior_precision_matrix == model.P₀

    #testing more complex dataset
    X, y = MLJBase.make_regression(100, 4; noise=0.5, sparse=0.2, outliers=0.1)
    #train, test = partition(eachindex(y), 0.7); # 70:30 split
    mach = MLJBase.machine(model, X, y)  
    MLJBase.fit!(mach; verbosity=0)
    yhat = MLJBase.predict(mach, X) # probabilistic predictions
    MLJBase.predict_mode(mach, X)   # point predictions
    MLJBase.fitted_params(mach)   #fitted params function 
    MLJBase.training_losses(mach) #training loss history
    model.epochs = 40 #changing number of epochs
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.epochs = 30 #changing number of epochs to a lower number
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.fit_prior_nsteps = 200 #changing LaplaceRedux fit steps
    MLJBase.fit!(mach; verbosity=0) #testing update function (the laplace part)
    yhat = MLJBase.predict(mach, X) # probabilistic predictions
    MLJBase.evaluate!(mach; resampling=cv, measure=SM.log_loss, verbosity=0)

    # Define a different model
    flux_model_two = Chain(Dense(4, 6, relu), Dense(6, 1))
    # test update! fallback to fit!
    model.model = flux_model_two
    MLJBase.fit!(mach; verbosity=0)
    model_two = LaplaceRegressor(; model=flux_model_two, epochs=100)
    @test !MLJBase.is_same_except(model, model_two)



    #testing default mlp builder
    model = LaplaceRegressor(; model=nothing, epochs=20)
    mach = MLJBase.machine(model, X, y)  
    MLJBase.fit!(mach; verbosity=1)
    yhat = MLJBase.predict(mach, X) # probabilistic predictions
    MLJBase.predict_mode(mach, X)   # point predictions
    MLJBase.fitted_params(mach)   #fitted params function 
    MLJBase.training_losses(mach) #training loss history
    model.epochs = 100 #changing number of epochs
    MLJBase.fit!(mach; verbosity=1) #testing update function

    #testing dataset_shape for one dimensional function
    X, y = MLJBase.make_regression(100, 1; noise=0.5, sparse=0.2, outliers=0.1)
    model = LaplaceRegressor(; model=nothing, epochs=20)
    mach = MLJBase.machine(model, X, y)  
    MLJBase.fit!(mach; verbosity=0)

    # applying generic interface tests from MLJTestInterface.jl
    Xraw, y = MLJTestInterface.make_regression()
    X = MLJBase.table(Float32.(MLJBase.matrix(Xraw)))
    y = Float32.(y)
    X1 = MLJBase.Tables.columntable(X)
    X2 = MLJBase.Tables.rowtable(X)
    datasets = [(X1, y), (X2, y)]
    @testset "LaplaceClassifier generic mlj interface tests" begin
        for data in datasets
            failures, summary = @test_logs(
                (:warn,),
                MLJTestInterface.test(
                [LaplaceRegressor,],
                data...;
                mod=@__MODULE__,
                verbosity=1, # bump to debug
                throw=true,  # set to true to debug
                ),
            )
            @test isempty(failures)
        end
    end

end

@testset "Classification" begin
    @info " testing  interface for LaplaceClassifier"
    # Define the model
    flux_model = Chain(Dense(4, 10, relu), Dense(10, 3))

    model = LaplaceClassifier(; model=flux_model, epochs=20)

    X, y = MLJBase.@load_iris
    mach = MLJBase.machine(model, X, y)
    MLJBase.fit!(mach; verbosity=0)
    Xnew = (
        sepal_length=[6.4, 7.2, 7.4],
        sepal_width=[2.8, 3.0, 2.8],
        petal_length=[5.6, 5.8, 6.1],
        petal_width=[2.1, 1.6, 1.9],
    )
    yhat = MLJBase.predict(mach, Xnew) # probabilistic predictions
    MLJBase.predict_mode(mach, Xnew)   # point predictions
    MLJBase.pdf.(yhat, "virginica")    # probabilities for the "verginica" class
    MLJBase.fitted_params(mach)  # fitted params 
    MLJBase.training_losses(mach) #training loss history
    model.epochs = 40 #changing number of epochs
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.epochs = 30 #changing number of epochs to a lower number
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.fit_prior_nsteps = 200 #changing LaplaceRedux fit steps
    MLJBase.fit!(mach; verbosity=0) #testing update function (the laplace part)
    MLJBase.evaluate!(mach; resampling=cv, measure=SM.brier_loss, verbosity=0)

    # Define a different model
    flux_model_two = Chain(Dense(4, 6, relu), Dense(6, 3))

    model.model = flux_model_two

    MLJBase.fit!(mach; verbosity=0)

    # applying generic interface tests from MLJTestInterface.jl
    Xraw, y = MLJTestInterface.make_multiclass()
    X = MLJBase.table(Float32.(MLJBase.matrix(Xraw)))
    X1 = MLJBase.Tables.columntable(X)
    X2 = MLJBase.Tables.rowtable(X)
    datasets = [(X1, y), (X2, y)]
    @testset "LaplaceClassifier generic mlj interface tests" begin
        for data in datasets
            failures, summary = @test_logs(
                (:warn,),
                MLJTestInterface.test(
                    [LaplaceClassifier,],
                    data...;
                    mod=@__MODULE__,
                    verbosity=1, # bump to debug
                    throw=true,  # set to true to debug
                ),
            )
            @test isempty(failures)
        end
    end

end
