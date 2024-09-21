using Flux
using Random
using Tables
using LinearAlgebra
using LaplaceRedux
using MLJBase
import MLJModelInterface as MMI
using Distributions: Normal

"""
    MLJBase.@mlj_model mutable struct LaplaceRegressor <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace regression model.
It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
It has the following Hyperparameters:
- `flux_model`: A Flux model provided by the user and compatible with the dataset.
- `flux_loss` : a Flux loss function
- `optimiser` = a Flux optimiser
- `epochs`: The number of training epochs.
- `batch_size`: The batch size.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
"""
MLJBase.@mlj_model mutable struct LaplaceRegressor <: MLJFlux.MLJFluxProbabilistic
    flux_model::Flux.Chain = nothing
    flux_loss = Flux.Losses.mse
    optimiser = Adam()
    epochs::Integer = 1000::(_ > 0)
    batch_size::Integer = 32::(_ > 0)
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    fit_prior_nsteps::Int = 100::(_ > 0)
end

@doc """
    MMI.fit(m::LaplaceRegressor, verbosity, X, y)

Fit a LaplaceRegressor model using the provided features and target values.

# Arguments
- `m::LaplaceRegressor`: The LaplaceRegressor model to be fitted.
- `verbosity`: Verbosity level for logging.
- `X`: Input features, expected to be in a format compatible with MLJBase.matrix.
- `y`: Target values.

# Returns
- `fitresult`: The fitted Laplace model.
- `cache`: Currently unused, returns `nothing`.
- `report`: A tuple containing the status and message of the fitting process.

# Description
This function performs the following steps:
1. Converts the input features `X` to a matrix and transposes it.
2. Reshapes the target values `y` to shape (1,:).
3. Creates a data loader for batching the data.
4. Sets up the optimizer state using the Adam optimizer.
5. Trains the model for a specified number of epochs.
6. Initializes a Laplace model with the trained Flux model and specified parameters.
7. Fits the Laplace model using the data loader.
8. Optimizes the prior of the Laplace model.
9. Returns the fitted Laplace model, a cache (currently `nothing`), and a report indicating success.
"""
function MMI.fit(m::LaplaceRegressor, verbosity, X, y)

    X = MLJBase.matrix(X) |> permutedims
    y = reshape(y, 1, :)
    data_loader = Flux.DataLoader((X, y); batchsize=m.batch_size)
    opt_state = Flux.setup(m.optimiser(), m.flux_model)

    for epoch in 1:(m.epochs)
        Flux.train!(m.flux_model, data_loader, opt_state) do model, X, y
            m.flux_loss(model(X), y)
        end
    end

    la = LaplaceRedux.Laplace(
        m.flux_model;
        likelihood=:regression,
        subset_of_weights=m.subset_of_weights,
        subnetwork_indices=m.subnetwork_indices,
        hessian_structure=m.hessian_structure,
        backend=m.backend,
        σ=m.σ,
        μ₀=m.μ₀,
        P₀=m.P₀,
    )

    # fit the Laplace model:
    LaplaceRedux.fit!(la, data_loader)
    optimize_prior!(la; verbose=false, n_steps=m.fit_prior_nsteps)

    fitresult = la
    report = (status="success", message="Model fitted successfully")
    cache = nothing
    return (fitresult, cache, report)
end

@doc """ 
function MMI.predict(m::LaplaceRegressor, fitresult, Xnew)

 Predicts the response for new data using a fitted LaplaceRegressor model.

 # Arguments
 - `m::LaplaceRegressor`: The LaplaceRegressor model.
 - `fitresult`: The result of fitting the LaplaceRegressor model.
 - `Xnew`: The new data for which predictions are to be made.

 # Returns
 - An array of Normal distributions, each centered around the predicted mean and variance for the corresponding input in `Xnew`.

 The function first converts `Xnew` to a matrix and permutes its dimensions. It then uses the `LaplaceRedux.predict` function to obtain the predicted means and variances. 
Finally, it creates Normal distributions from these means and variances and returns them as an array.
"""
function MMI.predict(m::LaplaceRegressor, fitresult, Xnew)
    Xnew = MLJBase.matrix(Xnew) |> permutedims
    la = fitresult
    yhat = LaplaceRedux.predict(la, Xnew; ret_distr=false)
    # Extract mean and variance matrices
    means, variances = yhat

    # Create Normal distributions from the means and variances
    return [Normal(μ, sqrt(σ)) for (μ, σ) in zip(means, variances)]
end

"""
    MLJBase.@mlj_model mutable struct LaplaceClassifier <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace Classification model.
It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
The model also has the following parameters:

- `flux_model`: A Flux model provided by the user and compatible with the dataset.
- `flux_loss` : a Flux loss function
- `optimiser` = a Flux optimiser
- `epochs`: The number of training epochs.
- `batch_size`: The batch size.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `link_approx`: the link approximation to use, either `:probit` or `:plugin`.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
"""
MLJBase.@mlj_model mutable struct LaplaceClassifier <: MLJFlux.MLJFluxProbabilistic
    flux_model::Flux.Chain = nothing
    flux_loss = Flux.Losses.logitcrossentropy
    optimiser = Adam()
    epochs::Integer = 1000::(_ > 0)
    batch_size::Integer = 32::(_ > 0)
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    fit_prior_nsteps::Int = 100::(_ > 0)
    link_approx::Symbol = :probit::(_ in (:probit, :plugin))
end

@doc """ 

 function MMI.fit(m::LaplaceClassifier, verbosity, X, y)
 
 Description:
 This function fits a LaplaceClassifier model using the provided data. It first preprocesses the input data `X` and target labels `y`, 
 then trains a neural network model using the Flux library. After training, it fits a Laplace approximation to the trained model.
 
 Arguments:
 - `m::LaplaceClassifier`: The LaplaceClassifier model to be fitted.
 - `verbosity`: Verbosity level for logging.
 - `X`: Input data features.
 - `y`: Target labels.
 
 Returns:
 - A tuple containing:
   - `(la, decode)`: The fitted Laplace model and the decode function for the target labels.
   - `cache`: A placeholder for any cached data (currently `nothing`).
   - `report`: A report dictionary containing the status and message of the fitting process.
 
 Notes:
 - The function uses the Flux library for neural network training and the LaplaceRedux library for fitting the Laplace approximation.
 - The `optimize_prior!` function is called to optimize the prior parameters of the Laplace model.

"""
function MMI.fit(m::LaplaceClassifier, verbosity, X, y)
    X = MLJBase.matrix(X) |> permutedims
    decode = y[1]
    y_plain = MLJBase.int(y) .- 1
    y_onehot = Flux.onehotbatch(y_plain, unique(y_plain))
    data_loader = Flux.DataLoader((X, y_onehot); batchsize=m.batch_size)
    opt_state = Flux.setup(m.optimiser, m.flux_model)

    for epoch in 1:(m.epochs)
        Flux.train!(m.flux_model, data_loader, opt_state) do model, X, y_onehot
            m.flux_loss(model(X), y_onehot)
        end
    end

    la = LaplaceRedux.Laplace(
        m.flux_model;
        likelihood=:classification,
        subset_of_weights=m.subset_of_weights,
        subnetwork_indices=m.subnetwork_indices,
        hessian_structure=m.hessian_structure,
        backend=m.backend,
        σ=m.σ,
        μ₀=m.μ₀,
        P₀=m.P₀,
    )

    # fit the Laplace model:
    LaplaceRedux.fit!(la, data_loader)
    optimize_prior!(la; verbose=false, n_steps=m.fit_prior_nsteps)

    report = (status="success", message="Model fitted successfully")
    cache = nothing
    return ((la, decode), cache, report)
end

@doc """ 
Predicts the class probabilities for new data using a Laplace classifier.

 # Arguments
 - `m::LaplaceClassifier`: The Laplace classifier model.
 - `(fitresult, decode)`: A tuple containing the fitted model result and the decode function.
 - `Xnew`: The new data for which predictions are to be made.

 # Returns
 - `MLJBase.UnivariateFinite`: The predicted class probabilities for the new data.

The function transforms the new data `Xnew` into a matrix, applies the LaplaceRedux
prediction function, and then returns the predictions as a `MLJBase.UnivariateFinite` object.
"""
function MMI.predict(m::LaplaceClassifier, (fitresult, decode), Xnew)
    la = fitresult
    Xnew = MLJBase.matrix(Xnew) |> permutedims
    predictions =
        LaplaceRedux.predict(la, Xnew; link_approx=m.link_approx, ret_distr=false) |>
        permutedims

    return MLJBase.UnivariateFinite(MLJBase.classes(decode), predictions)
end

MLJBase.metadata_model(
    LaplaceClassifier;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite,MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Continuous), # table with mixed types
    },
    target_scitype=AbstractArray{<:MLJBase.Finite}, # ordered factor or multiclass
    load_path="LaplaceRedux.LaplaceClassifier",
)
# metadata for each model,
MLJBase.metadata_model(
    LaplaceRegressor;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite,MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Continuous), # table with mixed types
    },
    target_scitype=AbstractArray{MLJBase.Continuous},
    load_path="LaplaceRedux.LaplaceRegressor",
)
