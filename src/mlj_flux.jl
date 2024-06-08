using Flux
using MLJFlux
using ProgressMeter: Progress, next!, BarGlyphs
using Random
using Tables
using Distributions
using LinearAlgebra
using LaplaceRedux
import MLJBase
import MLJBase: @mlj_model, metadata_model, metadata_pkg

"""
    @mlj_model mutable struct LaplaceClassification <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace Classification model that extends the MLJFluxProbabilistic abstract type.
    It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
    The model is trained using the `fit!` method. The model is defined by the following default parameters:

- `loss`: a loss function that takes the predicted output and the true output as arguments.
- `rng`: a random number generator.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `link_approx`: the link approximation to use, either `:probit` or `:plugin`.
- `predict_proba`:whether to compute the probabilities or not, either true or false.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
- `la`: The fitted Laplace object will be saved here once the model is fitted. It has to be left to nothing, the fit! function will automatically save the Laplace model here.
"""
@mlj_model mutable struct LaplaceClassification <: MLJFlux.MLJFluxProbabilistic
    loss=Flux.crossentropy 
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG 
    subset_of_weights::Symbol=:all 
    subnetwork_indices::Vector{Vector{Int}}=Vector{Vector{Int}}([])
    hessian_structure::Union{HessianStructure,Symbol,String}=:full
    backend::Symbol=:GGN 
    σ::Float64=1.0 
    μ₀::Float64=0.0 
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing
    link_approx::Symbol=:probit 
    predict_proba::Bool= true 
    fit_prior_nsteps::Int=100 
    la::Union{Nothing,AbstractLaplace}= nothing
    
end

"""
    @mlj_model mutable struct LaplaceRegression <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace regression model that extends the `MLJFlux.MLJFluxProbabilistic` abstract type.
It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
The model is trained using the `fit!` method. The model is defined by the following default parameters:

- `loss`: a loss function that takes the predicted output and the true output as arguments.
- `rng`: a random number generator.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
- `la`: The fitted Laplace object will be saved here once the model is fitted. It has to be left to nothing, the fit! function will automatically save the Laplace model here.
"""
@mlj_model mutable struct LaplaceRegression <: MLJFlux.MLJFluxProbabilistic
    loss=Flux.Losses.mse 
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG
    subset_of_weights::Symbol=:all
    subnetwork_indices=nothing
    hessian_structure::Union{HessianStructure,Symbol,String}=:full
    backend::Symbol=:GGN
    σ::Float64=1.0
    μ₀::Float64=0.0 
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing
    fit_prior_nsteps::Int=100
    la::Union{Nothing,AbstractLaplace}= nothing
end
const MLJ_Laplace= Union{LaplaceClassification,LaplaceRegression}

 

"""
    MLJFlux.fit!(model::LaplaceRegression, penalty, chain, epochs, batch_size, optimiser, verbosity, X, y)

Fit the LaplaceRegression model using Flux.jl.

# Arguments
- `model::LaplaceRegression`: The LaplaceRegression object.
- `penalty`: The penalty term for regularization.
- `chain`: The chain of layers for the model.
- `epochs`: The number of training epochs.
- `batch_size`: The size of each training batch.
- `optimiser`: The optimization algorithm to use.
- `verbosity`: The level of verbosity during training.
- `X`: The input data.
- `y`: The target data.

# Returns
- `model::LaplaceRegression`: The fitted LaplaceRegression model.
"""
function MLJFlux.fit!(model::LaplaceRegression, penalty,chain, epochs,batch_size, optimiser, verbosity, X, y)

    X = MLJBase.matrix(X, transpose=true) 
    model.la = Laplace(
        chain;
        likelihood=:regression,
        subset_of_weights=model.subset_of_weights,
        subnetwork_indices=model.subnetwork_indices,
        hessian_structure=model.hessian_structure,
        backend=model.backend,
        σ=model.σ,
        μ₀=model.μ₀,
        P₀=model.P₀)
    n_samples= size(X,1)
    # Initialize history:
    history = []
    verbose_laplace=false
    # Define the loss function for Laplace Regression with a custom penalty
    function custom_loss( y_pred, y_batch)
        data_loss = model.loss( y_pred,y_batch)
        penalty_term = penalty(Flux.params(chain))
        return data_loss + penalty_term
    end
    # intitialize and start progress meter:
    meter = Progress(
        epochs + 1;
        dt=1.0,
        desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=25,
        color=:yellow,
    )
    # Create a data loader
    loader = Flux.Data.DataLoader((data=X, label=y), batchsize=batch_size, shuffle=true)
    parameters = Flux.params(chain)
    for i in 1:epochs
        epoch_loss = 0.0
        # train the model
        for (X_batch, y_batch) in loader
            y_batch = reshape(y_batch,1,:)
        
            # Backward pass
            gs = Flux.gradient(parameters) do
                batch_loss =  custom_loss(chain(X_batch), y_batch)
                epoch_loss += batch_loss
            end
            # Update parameters
            Flux.update!(optimiser, parameters,gs)
        end
        epoch_loss /= n_samples
        push!(history, epoch_loss)
        #verbosity
        if verbosity == 1
            next!(meter)
        elseif  verbosity ==2
            next!(meter)
            verbose_laplace=true
            println( "Loss is $(round(epoch_loss; sigdigits=4))")
        end
    end

 

    # fit the Laplace model:
    LaplaceRedux.fit!(model.la,zip(eachcol(X),y))
    optimize_prior!(model.la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)
    cache= nothing
    report=[]

    return (model, cache, report)
end


"""
    predict(model::LaplaceRegression, Xnew)

Predict the output for new input data using a Laplace regression model.

# Arguments
- `model::LaplaceRegression`: The trained Laplace regression model.
- `Xnew`: The new input data.

# Returns
- The predicted output for the new input data.

"""
function MLJFlux.predict(model::LaplaceRegression, Xnew)
    Xnew = MLJBase.matrix(Xnew) 
    #convert in a vector of vectors because MLJ ask to do so
    X_vec= [Xnew[i,:] for i in 1:size(Xnew, 1)]
    #inizialize output vector yhat
    yhat=[]
    # Predict using Laplace and collect the predictions
    yhat = [glm_predictive_distribution(model.la, x_vec) for x_vec in X_vec]

    predictions = []
    for row in eachrow(yhat)

        mean_val = Float64(row[1][1][1])
        std_val = sqrt(Float64(row[1][2][1]))
        # Append a Normal distribution:
        push!(predictions, Normal(mean_val, std_val))
    end
    
    return predictions

end





#########################################  fit and predict for classification


"""
    MLJFlux.fit!(model::LaplaceClassification, penalty, chain, epochs, batch_size, optimiser, verbosity, X, y)

Fit the LaplaceClassification model using MLJFlux.

# Arguments
- `model::LaplaceClassification`: The LaplaceClassification object to fit.
- `penalty`: The penalty to apply during training.
- `chain`: The chain to use during training.
- `epochs`: The number of training epochs.
- `batch_size`: The batch size for training.
- `optimiser`: The optimiser to use during training.
- `verbosity`: The verbosity level for training.
- `X`: The input data for training.
- `y`: The target labels for training.

# Returns
- `model::LaplaceClassification`: The fitted LaplaceClassification model.
"""
function MLJFlux.fit!(model::LaplaceClassification, penalty,chain, epochs,batch_size, optimiser, verbosity, X, y)
    X = MLJBase.matrix(X, transpose=true) 

    # Integer encode the target variable y
    #y_onehot = unique(y) .== permutedims(y)

    model.la = Laplace(
        chain;
        likelihood=:classification,
        subset_of_weights=model.subset_of_weights,
        subnetwork_indices=model.subnetwork_indices,
        hessian_structure=model.hessian_structure,
        backend=model.backend,
        σ=model.σ,
        μ₀=model.μ₀,
        P₀=model.P₀)
    n_samples= size(X,1)
    verbose_laplace=false

   # Initialize history:
   history = []
   # Define the loss function for Laplace Regression with a custom penalty
   function custom_loss( y_pred, y_batch)
    data_loss = model.loss( y_pred,y_batch)
    penalty_term = penalty(Flux.params(chain))
    return data_loss + penalty_term
    end
   # intitialize and start progress meter:
   meter = Progress(
       epochs + 1;
       dt=0,
       desc="Optimising neural net:",
       barglyphs=BarGlyphs("[=> ]"),
       barlen=25,
       color=:yellow,
   )
   # Create a data loader
   loader = Flux.Data.DataLoader((data=X, label=y), batchsize=batch_size, shuffle=true)
   parameters = Flux.params(chain)
   for i in 1:epochs
       epoch_loss = 0.0
       # train the model
       for (X_batch, y_batch) in loader

           # Backward pass
           gs = Flux.gradient(parameters) do
            batch_loss =  custom_loss(chain(X_batch), y_batch)
            epoch_loss += batch_loss
            end
           # Update parameters
           Flux.update!(optimiser, parameters,gs)
        end
       epoch_loss /= n_samples
       push!(history, epoch_loss)
       #verbosity
       if verbosity == 1
            next!(meter)
        elseif  verbosity ==2
            next!(meter)
            verbose_laplace=true
            println( "Loss is $(round(epoch_loss; sigdigits=4))")
        end
   end

   # fit the Laplace model:
   LaplaceRedux.fit!(model.la,zip(eachcol(X),y))
   optimize_prior!(model.la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)
   cache= nothing
   report=[]

   return (model, cache, report)
end


"""
    predict(model::LaplaceClassification, Xnew)

Predicts the class labels for new data using the LaplaceClassification model.

# Arguments
- `model::LaplaceClassification`: The trained LaplaceClassification model.
- `Xnew`: The new data to make predictions on.

# Returns
An array of predicted class labels.

"""
function MLJFlux.predict(model::LaplaceClassification, Xnew)
    Xnew = MLJBase.matrix(Xnew) 
    #convert in a vector of vectors because Laplace ask to do so
    X_vec= X_vec= [Xnew[i,:] for i in 1:size(Xnew, 1)]

    # Predict using Laplace and collect the predictions
    predictions = [LaplaceRedux.predict(model.la, x;link_approx= model.link_approx,predict_proba=model.predict_proba) for x in X_vec]
    
    return predictions

end

# Then for each model,
MLJBase.metadata_model(
    LaplaceClassification;
    input=Union{
        AbstractMatrix{MLJBase.Continuous},
        MLJBase.Table(MLJBase.Continuous),
        MLJBase.Table{AbstractVector{MLJBase.Continuous}},
    },
    target=Union{
        AbstractArray{MLJBase.Finite},
        AbstractArray{MLJBase.Continuous},
        AbstractVector{MLJBase.Finite},
        AbstractVector{MLJBase.Continuous},
    },
    path="MLJFlux.LaplaceClassification",
)
# Then for each model,
MLJBase.metadata_model(
    LaplaceRegression;
    input=Union{
        AbstractMatrix{MLJBase.Continuous},
        MLJBase.Table(MLJBase.Continuous),
        MLJBase.Table{AbstractVector{MLJBase.Continuous}},
    },
    target=Union{
        AbstractArray{MLJBase.Finite},
        AbstractArray{MLJBase.Continuous},
        AbstractVector{MLJBase.Finite},
        AbstractVector{MLJBase.Continuous},
    },
    path="MLJFlux.LaplaceRegression",
)
 