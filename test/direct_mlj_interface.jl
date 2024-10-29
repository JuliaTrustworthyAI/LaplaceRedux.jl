using Random: Random
import Random.seed!
using MLJBase: MLJBase, categorical
using MLJ: MLJ
using Flux
using StableRNGs
import LaplaceRedux: LaplaceClassifier, LaplaceRegressor

cv = MLJBase.CV(; nfolds=3)

@testset "Regression" begin
    @info " testing  interface for LaplaceRegressor"
    flux_model = Chain(Dense(4, 10, relu), Dense(10, 10, relu), Dense(10, 1))
    model = LaplaceRegressor(; model=flux_model, epochs=50)

    X, y = MLJ.make_regression(100, 4; noise=0.5, sparse=0.2, outliers=0.1)
    #train, test = partition(eachindex(y), 0.7); # 70:30 split
    mach = MLJ.machine(model, X, y)  
    MLJBase.fit!(mach; verbosity=0)
    yhat = MLJBase.predict(mach, X) # probabilistic predictions
    MLJBase.predict_mode(mach, X)   # point predictions
    MLJBase.fitted_params(mach)   #fitted params function 
    MLJBase.training_losses(mach) #training loss history
    model.epochs = 100 #changing number of epochs
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.epochs = 50 #changing number of epochs to a lower number
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.fit_prior_nsteps = 200 #changing LaplaceRedux fit steps
    MLJBase.fit!(mach; verbosity=0) #testing update function (the laplace part)
    yhat = MLJBase.predict(mach, X) # probabilistic predictions
    MLJ.evaluate!(mach; resampling=cv, measure=MLJ.log_loss, verbosity=0)

    # Define a different model
    flux_model_two = Chain(Dense(4, 6, relu), Dense(6, 1))
    # test update! fallback to fit!
    model.model = flux_model_two
    MLJBase.fit!(mach; verbosity=0)

    model_two = LaplaceRegressor(; model=flux_model_two, epochs=100)
    @test !MLJBase.is_same_except(model, model_two)


    #testing default mlp builder
    model = LaplaceRegressor(; model=nothing, epochs=50)
    mach = MLJ.machine(model, X, y)  
    MLJBase.fit!(mach; verbosity=0)
    yhat = MLJBase.predict(mach, X) # probabilistic predictions
    MLJBase.predict_mode(mach, X)   # point predictions
    MLJBase.fitted_params(mach)   #fitted params function 
    MLJBase.training_losses(mach) #training loss history
    model.epochs = 100 #changing number of epochs
    MLJBase.fit!(mach; verbosity=0) #testing update function



end

@testset "Classification" begin
    @info " testing  interface for LaplaceClassifier"
    # Define the model
    flux_model = Chain(Dense(4, 10, relu), Dense(10, 3))

    model = LaplaceClassifier(; model=flux_model, epochs=50)

    X, y = MLJ.@load_iris
    mach = MLJ.machine(model, X, y)
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
    model.epochs = 100 #changing number of epochs
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.epochs = 50 #changing number of epochs to a lower number
    MLJBase.fit!(mach; verbosity=0) #testing update function
    model.fit_prior_nsteps = 200 #changing LaplaceRedux fit steps
    MLJBase.fit!(mach; verbosity=0) #testing update function (the laplace part)
    MLJ.evaluate!(mach; resampling=cv, measure=MLJ.brier_loss, verbosity=0)

    # Define a different model
    flux_model_two = Chain(Dense(4, 6, relu), Dense(6, 3))

    model.model = flux_model_two

    MLJBase.fit!(mach; verbosity=0)
end
