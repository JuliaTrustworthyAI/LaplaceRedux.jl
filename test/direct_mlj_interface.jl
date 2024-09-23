using Random: Random
import Random.seed!
using MLJBase: MLJBase, categorical
using Flux
using StableRNGs
using MLJ
using LaplaceRedux


@testset "Regression" begin
    flux_model = Chain(
        Dense(4, 10, relu),
        Dense(10, 10, relu),
        Dense(10, 1)
    )
    model = LaplaceRegressor(model=flux_model,epochs=50)
    
    X, y = make_regression(100, 4; noise=0.5, sparse=0.2, outliers=0.1)
    mach = machine(model, X, y) #|> MLJBase.fit! #|> (fitresult,cache,report)
    MLJBase.fit!(mach)
    Xnew, _ = make_regression(3, 4; rng=123)
    yhat = MLJBase.predict(mach, Xnew) # probabilistic predictions
    MLJBase.predict_mode(mach, Xnew)   # point predictions
    MLJBase.fitted_params(mach)   #fitted params function 
end



@testset "Classification" begin
# Define the model
flux_model = Chain(
    Dense(4, 10, relu),
    Dense(10, 10, relu),
    Dense(10, 3)
)

model = LaplaceClassifier(model=flux_model,epochs=50)

X, y = @load_iris
mach = machine(model, X, y)
MLJBase.fit!(mach)
Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = MLJBase.predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class
MLJBase.fitted_params(mach)  # fitted params 
   
end
