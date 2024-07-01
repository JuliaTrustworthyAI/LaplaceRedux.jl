using Flux
using MLJFlux
using ProgressMeter: Progress, next!, BarGlyphs
using Random
using Tables
using LinearAlgebra
using LaplaceRedux
using ComputationalResources
using MLJBase: MLJBase
import MLJModelInterface as MMI
using Optimisers: Optimisers

"""
    MLJBase.@mlj_model mutable struct LaplaceRegression <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace regression model that extends the `MLJFlux.MLJFluxProbabilistic` abstract type.
It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
The model is trained using the `fit!` method. The model is defined by the following default parameters:

- `builder`: a Flux model that constructs the neural network.
- `optimiser`: a Flux optimiser.
- `loss`: a loss function that takes the predicted output and the true output as arguments.
- `epochs`: the number of epochs.
- `batch_size`: the size of a batch.
- `lambda`: the regularization strength.
- `alpha`: the regularization mix (0 for all l2, 1 for all l1).
- `rng`: a random number generator.
- `optimiser_changes_trigger_retraining`: a boolean indicating whether changes in the optimiser trigger retraining.
- `acceleration`: the computational resource to use.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
"""
MLJBase.@mlj_model mutable struct LaplaceRegression <: MLJFlux.MLJFluxProbabilistic
    builder = MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish)
    optimiser = Optimisers.Adam()
    loss = Flux.Losses.mse
    epochs::Int = 10::(_ > 0)
    batch_size::Int = 1::(_ > 0)
    lambda::Float64 = 1.0
    alpha::Float64 = 0.0
    rng::Union{AbstractRNG,Int64} = Random.GLOBAL_RNG
    optimiser_changes_trigger_retraining::Bool = false::(_ in (true, false))
    acceleration = CPU1()::(_ in (CPU1(), CUDALibs()))
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

"""
    MLJBase.@mlj_model mutable struct LaplaceClassification <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace Classification model that extends the MLJFluxProbabilistic abstract type.
    It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
    The model is trained using the `fit!` method. The model is defined by the following default parameters:

- `builder`: a Flux model that constructs the neural network.
- `finaliser`: a Flux model that processes the output of the neural network.
- `optimiser`: a Flux optimiser.
- `loss`: a loss function that takes the predicted output and the true output as arguments.
- `epochs`: the number of epochs.
- `batch_size`: the size of a batch.
- `lambda`: the regularization strength.
- `alpha`: the regularization mix (0 for all l2, 1 for all l1).
- `rng`: a random number generator.
- `optimiser_changes_trigger_retraining`: a boolean indicating whether changes in the optimiser trigger retraining.
- `acceleration`: the computational resource to use.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
"""
MLJBase.@mlj_model mutable struct LaplaceClassification <: MLJFlux.MLJFluxProbabilistic
    builder = MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish)
    finaliser = Flux.softmax
    optimiser = Optimisers.Adam()
    loss = Flux.crossentropy
    epochs::Int = 10::(_ > 0)
    batch_size::Int = 1::(_ > 0)
    lambda::Float64 = 1.0
    alpha::Float64 = 0.0
    rng::Union{AbstractRNG,Int64} = Random.GLOBAL_RNG
    optimiser_changes_trigger_retraining::Bool = false::(_ in (true, false))
    acceleration = CPU1()::(_ in (CPU1(), CUDALibs()))
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices::Vector{Vector{Int}} = Vector{Vector{Int}}([])
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    link_approx::Symbol = :probit::(_ in (:probit, :plugin))
    predict_proba::Bool = true::(_ in (true, false))
    fit_prior_nsteps::Int = 100::(_ > 0)
end

const MLJ_Laplace = Union{LaplaceClassification,LaplaceRegression}

"""
    MLJFlux.shape(model::LaplaceRegression, X, y)

Compute the the number of features of the X input dataset and  the number of variables to predict from  the y  output dataset.

# Arguments
- `model::LaplaceRegression`: The LaplaceRegression model to fit.
- `X`: The input data for training.
- `y`: The target labels for training one-hot encoded.

# Returns
- (input size, output size)
"""
function MLJFlux.shape(model::LaplaceRegression, X, y)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X
    n_input = size(X, 2)
    dims = size(y)
    if length(dims) == 1
        n_output = 1
    else
        n_output = dims[1]
    end
    return (n_input, n_output)
end

"""
    MLJFlux.build(model::LaplaceRegression, rng, shape)

Builds an MLJFlux model for Laplace regression compatible with the dimensions of the input and output layers specified by `shape`.

# Arguments
- `model::LaplaceRegression`: The Laplace regression model.
- `rng`: A random number generator to ensure reproducibility.
- `shape`: A tuple or array specifying the dimensions of the input and output layers.

# Returns
- The constructed MLJFlux model, compatible with the specified input and output dimensions.
"""
function MLJFlux.build(model::LaplaceRegression, rng, shape)
    chain = MLJFlux.build(model.builder, rng, shape...)
    return chain
end

"""
    MLJFlux.fitresult(model::LaplaceRegression, chain, y)

Computes the fit result for a Laplace Regression model, returning the model chain and the number of output variables in the target data.

# Arguments
- `model::LaplaceRegression`: The Laplace Regression model to be evaluated.
- `chain`: The trained model chain.
- `y`: The target data, typically a vector of class labels.

# Returns
- A tuple containing:
  - The model.
  - The trained Flux chain.
  - The number of unique classes in the target data `y`.
"""
function MLJFlux.fitresult(model::LaplaceRegression, chain, y)
    if y isa AbstractArray
        target_column_names = nothing
    else
        target_column_names = Tables.schema(y).names
    end
    return (chain, deepcopy(model))
end

"""
    MLJFlux.train(model::LaplaceRegression, chain, regularized_optimiser, optimiser_state, epochs, verbosity, X, y)

Fit the LaplaceRegression model using Flux.jl.

# Arguments
- `model::LaplaceRegression`: The LaplaceRegression model.
- `regularized_optimiser`: the regularized optimiser to apply to the loss function.
- `optimiser_state`: thestate of the optimiser.
- `epochs`: The number of epochs for training.
- `verbosity`: The verbosity level for training.
- `X`: The input data for training.
- `y`: The target labels for training.

# Returns (fitresult, cache, report )
where
- `fitresult`: is the output of MLJFlux.fitresult.
- `cache`: an empty tuple.
- `report`: a named tuple that contain the field training_losses.
"""
function MLJFlux.train(
    model::LaplaceRegression,
    chain,
    regularized_optimiser,
    optimiser_state,
    epochs,
    verbosity,
    X,
    y,
)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X

    if !isa(chain, AbstractLaplace)
        la = LaplaceRedux.Laplace(
            chain;
            likelihood=:regression,
            subset_of_weights=model.subset_of_weights,
            subnetwork_indices=model.subnetwork_indices,
            hessian_structure=model.hessian_structure,
            backend=model.backend,
            σ=model.σ,
            μ₀=model.μ₀,
            P₀=model.P₀,
        )
    else
        la = chain
    end

    # Initialize history:
    history = []
    verbose_laplace = false
    # intitialize and start progress meter:
    meter = Progress(
        epochs + 1;
        dt=1.0,
        desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=25,
        color=:yellow,
    )
    verbosity != 1 || next!(meter)

    # initiate history:
    loss = model.loss
    n_batches = length(y)
    losses = (loss(chain(X[i]), y[i]) for i in 1:n_batches)
    history = [mean(losses)]

    for i in 1:epochs
        chain, optimiser_state, current_loss = MLJFlux.train_epoch(
            model, chain, regularized_optimiser, optimiser_state, X, y
        )
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    # fit the Laplace model:
    LaplaceRedux.fit!(la, zip(X, y))
    optimize_prior!(la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)

    return la, optimiser_state, history
end

"""
    predict(model::LaplaceRegression, Xnew)

Predict the output for new input data using a Laplace regression model.

# Arguments
- `model::LaplaceRegression`: The trained Laplace regression model.
- the fitresult output produced by MLJFlux.fit!
- `Xnew`: The new input data.

# Returns
- The predicted output for the new input data.

"""
function MLJFlux.predict(model::LaplaceRegression, fitresult, Xnew)
    Xnew = MLJBase.matrix(Xnew)

    model = fitresult[1]
    #convert in a vector of vectors because MLJ ask to do so
    X_vec = [Xnew[i, :] for i in 1:size(Xnew, 1)]
    #inizialize output vector yhat
    yhat = []
    # Predict using Laplace and collect the predictions
    yhat = [glm_predictive_distribution(model, x_vec) for x_vec in X_vec]

    return yhat
end

"""
    MLJFlux.shape(model::LaplaceClassification, X, y)

Compute the the number of features of the dataset X and  the number of unique classes in y.

# Arguments
- `model::LaplaceClassification`: The LaplaceClassification model to fit.
- `X`: The input data for training.
- `y`: The target labels for training one-hot encoded.

# Returns
- (input size, output size)
"""

function MLJFlux.shape(model::LaplaceClassification, X, y)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X
    n_input = size(X, 2)
    levels = unique(y)
    n_output = length(levels)
    return (n_input, n_output)
end

"""
    MLJFlux.build(model::LaplaceClassification, rng, shape)

Builds an MLJFlux model for Laplace classification compatible with the dimensions of the input and output layers specified by `shape`.

# Arguments
- `model::LaplaceClassification`: The Laplace classification model.
- `rng`: A random number generator to ensure reproducibility.
- `shape`: A tuple or array specifying the dimensions of the input and output layers.

# Returns
- The constructed MLJFlux model, compatible with the specified input and output dimensions.
"""
function MLJFlux.build(model::LaplaceClassification, rng, shape)
    chain = Flux.Chain(MLJFlux.build(model.builder, rng, shape...), model.finaliser)

    return chain
end

"""
    MLJFlux.fitresult(model::LaplaceClassification, chain, y)

Computes the fit result for a Laplace classification model, returning the model chain and the number of unique classes in the target data.

# Arguments
- `model::LaplaceClassification`: The Laplace classification model to be evaluated.
- `chain`: The trained model chain.
- `y`: The target data, typically a vector of class labels.

# Returns
- A tuple containing:
  - The model.
  - The number of unique classes in the target data `y`.
"""
function MLJFlux.fitresult(model::LaplaceClassification, chain, y)
    return (chain, deepcopy(model))
end

"""
    MLJFlux.train(model::LaplaceClassification, chain, regularized_optimiser, optimiser_state, epochs, verbosity, X, y)

Fit the LaplaceRegression model using Flux.jl.

# Arguments
- `model::LaplaceClassification`: The LaplaceClassification model.
- `regularized_optimiser`: the regularized optimiser to apply to the loss function.
- `optimiser_state`: thestate of the optimiser.
- `epochs`: The number of epochs for training.
- `verbosity`: The verbosity level for training.
- `X`: The input data for training.
- `y`: The target labels for training.

# Returns (fitresult, cache, report )
where
- `fitresult`: is the output of MLJFlux.fitresult.
- `cache`: an empty tuple.
- `report`: a named tuple that contain the field training_losses.
"""
function MLJFlux.train(
    model::LaplaceClassification,
    chain,
    regularized_optimiser,
    optimiser_state,
    epochs,
    verbosity,
    X,
    y,
)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X

    if !isa(chain, AbstractLaplace)
        la = LaplaceRedux.Laplace(
            chain;
            likelihood=:classification,
            subset_of_weights=model.subset_of_weights,
            subnetwork_indices=model.subnetwork_indices,
            hessian_structure=model.hessian_structure,
            backend=model.backend,
            σ=model.σ,
            μ₀=model.μ₀,
            P₀=model.P₀,
        )
    else
        la = chain
    end

    # Initialize history:
    history = []
    verbose_laplace = false
    # intitialize and start progress meter:
    meter = Progress(
        epochs + 1;
        dt=1.0,
        desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=25,
        color=:yellow,
    )
    verbosity != 1 || next!(meter)

    # initiate history:
    loss = model.loss
    n_batches = length(y)
    losses = (loss(chain(X[i]), y[i]) for i in 1:n_batches)
    history = [mean(losses)]

    for i in 1:epochs
        chain, optimiser_state, current_loss = MLJFlux.train_epoch(
            model, chain, regularized_optimiser, optimiser_state, X, y
        )
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    # fit the Laplace model:
    LaplaceRedux.fit!(la, zip(X, y))
    optimize_prior!(la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)

    return la, optimiser_state, history
end

"""
    predict(model::LaplaceClassification, Xnew)

Predicts the class labels for new data using the LaplaceClassification model.

# Arguments
- `model::LaplaceClassification`: The trained LaplaceClassification model.
- fitresult: the fitresult output produced by MLJFlux.fit!
- `Xnew`: The new data to make predictions on.

# Returns
An array of predicted class labels.

"""
function MLJFlux.predict(model::LaplaceClassification, fitresult, Xnew)
    la = fitresult[1]
    Xnew = MLJBase.matrix(Xnew)
    #convert in a vector of vectors because Laplace ask to do so
    X_vec = X_vec = [Xnew[i, :] for i in 1:size(Xnew, 1)]

    # Predict using Laplace and collect the predictions
    predictions = [
        LaplaceRedux.predict(
            la, x; link_approx=model.link_approx, predict_proba=model.predict_proba
        ) for x in X_vec
    ]

    return predictions
end

# metadata for each model,
MLJBase.metadata_model(
    LaplaceClassification;
    input=Union{
        AbstractMatrix{MLJBase.Finite},
        MLJBase.Table(MLJBase.Finite),
        AbstractMatrix{MLJBase.Continuous},
        MLJBase.Table(MLJBase.Continuous),
        MLJBase.Table{AbstractVector{MLJBase.Continuous}},
        MLJBase.Table{AbstractVector{MLJBase.Finite}},
    },
    target=Union{AbstractArray{MLJBase.Finite},AbstractArray{MLJBase.Continuous}},
    path="MLJFlux.LaplaceClassification",
)
# metadata for each model,
MLJBase.metadata_model(
    LaplaceRegression;
    input=Union{
        AbstractMatrix{MLJBase.Continuous},
        MLJBase.Table(MLJBase.Continuous),
        AbstractMatrix{MLJBase.Finite},
        MLJBase.Table(MLJBase.Finite),
        MLJBase.Table{AbstractVector{MLJBase.Continuous}},
        MLJBase.Table{AbstractVector{MLJBase.Finite}},
    },
    target=Union{AbstractArray{MLJBase.Finite},AbstractArray{MLJBase.Continuous}},
    path="MLJFlux.LaplaceRegression",
)
