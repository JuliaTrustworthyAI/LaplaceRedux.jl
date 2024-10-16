using Random: Random
import Random.seed!
using MLJBase: MLJBase, categorical
using Flux
using StableRNGs
using MLJ
using MLJ:predict,fit!
using LaplaceRedux

cv = CV(; nfolds=3)

@testset "Regression" begin
    flux_model = Chain(
        Dense(4, 10, relu),
        Dense(10, 10, relu),
        Dense(10, 1)
    )
    model = LaplaceRegressor(model=flux_model,epochs=50)
    
    X, y = make_regression(100, 4; noise=0.5, sparse=0.2, outliers=0.1)
    mach = machine(model, X, y) #|> MLJBase.fit! #|> (fitresult,cache,report)
    MLJBase.fit!(mach,verbosity=1)
    Xnew, _ = make_regression(3, 4; rng=123)
    yhat = MLJBase.predict(mach, Xnew) # probabilistic predictions
    MLJBase.predict_mode(mach, Xnew)   # point predictions
    MLJBase.fitted_params(mach)   #fitted params function 
    MLJBase.training_losses(mach) #training loss history
    model.epochs= 100 #changing number of epochs
    MLJBase.fit!(mach) #testing update function
    model.epochs= 50 #changing number of epochs to a lower number
    MLJBase.fit!(mach) #testing update function
    model.fit_prior_nsteps = 200 #changing LaplaceRedux fit steps
    MLJBase.fit!(mach) #testing update function (the laplace part)
    # evaluate!(mach, resampling=cv, measure=l2, verbosity=0)
end



@testset "Classification" begin
    # Define the model
    flux_model = Chain(
        Dense(4, 10, relu),
        Dense(10, 3)
    )

    model = LaplaceClassifier(model=flux_model,epochs=50)

    X, y = @load_iris
    mach = machine(model, X, y)
    MLJBase.fit!(mach,verbosity=1)
    Xnew = (sepal_length = [6.4, 7.2, 7.4],
            sepal_width = [2.8, 3.0, 2.8],
            petal_length = [5.6, 5.8, 6.1],
            petal_width = [2.1, 1.6, 1.9],)
    yhat = MLJBase.predict(mach, Xnew) # probabilistic predictions
    predict_mode(mach, Xnew)   # point predictions
    pdf.(yhat, "virginica")    # probabilities for the "verginica" class
    MLJBase.fitted_params(mach)  # fitted params 
    MLJBase.training_losses(mach) #training loss history
    model.epochs= 100 #changing number of epochs
    MLJBase.fit!(mach) #testing update function
    model.epochs= 50 #changing number of epochs to a lower number
    MLJBase.fit!(mach) #testing update function
    model.fit_prior_nsteps = 200 #changing LaplaceRedux fit steps
    MLJBase.fit!(mach) #testing update function (the laplace part)
    evaluate!(mach; resampling=cv, measure=brier_loss, verbosity=0)

    # Define a different model
    flux_model_two = Chain(
        Dense(4, 6, relu),
        Dense(6, 3)
    )

    model.model = flux_model_two

    MLJBase.fit!(mach)
end
