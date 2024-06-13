using Flux
using MLJFlux
using ProgressMeter: Progress, next!, BarGlyphs
using Random
using Tables
using Distributions
using LinearAlgebra
using LaplaceRedux
using ComputationalResources
using MLJBase
import MLJBase: @mlj_model, metadata_model, metadata_pkg


"""
    @mlj_model mutable struct LaplaceRegression <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace regression model that extends the `MLJFlux.MLJFluxProbabilistic` abstract type.
It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
The model is trained using the `fit!` method. The model is defined by the following default parameters:

- `builder`: a Flux model that constructs the neural network.
- `optimiser`: a Flux optimiser.
- `loss`: a loss function that takes the predicted output and the true output as arguments.
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
@mlj_model mutable struct LaplaceRegression <: MLJFlux.MLJFluxProbabilistic
    builder=MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish)
    optimiser= Flux.Optimise.Adam()
    loss=Flux.Losses.mse
    epochs::Int=10::(_ > 0) 
    batch_size::Int=1::(_ > 0)
    lambda::Float64=1.0
    alpha::Float64=0.0
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG
    optimiser_changes_trigger_retraining::Bool=false::(_ in (true, false))
    acceleration=CPU1()::(_ in (CPU1(), CUDALibs()))
    subset_of_weights::Symbol=:all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices=nothing
    hessian_structure::Union{HessianStructure,Symbol,String}=:full::(_ in (":full", ":diagonal"))
    backend::Symbol=:GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64=1.0
    μ₀::Float64=0.0 
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing
    fit_prior_nsteps::Int=100::(_ > 0)
end



"""
    @mlj_model mutable struct LaplaceClassification <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace Classification model that extends the MLJFluxProbabilistic abstract type.
    It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 
    The model is trained using the `fit!` method. The model is defined by the following default parameters:

- `builder`: a Flux model that constructs the neural network.
- `finaliser`: a Flux model that processes the output of the neural network.
- `optimiser`: a Flux optimiser.
- `loss`: a loss function that takes the predicted output and the true output as arguments.
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
@mlj_model mutable struct LaplaceClassification <: MLJFlux.MLJFluxProbabilistic
    builder=MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish)
    finaliser=Flux.softmax
    optimiser= Flux.Optimise.Adam()
    loss=Flux.crossentropy 
    epochs::Int=10::(_ > 0)
    batch_size::Int=1::(_ > 0)
    lambda::Float64=1.0
    alpha::Float64=0.0
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG 
    optimiser_changes_trigger_retraining::Bool=false
    acceleration=CPU1()::(_ in (CPU1(), CUDALibs()))
    subset_of_weights::Symbol=:all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices::Vector{Vector{Int}}=Vector{Vector{Int}}([]) 
    hessian_structure::Union{HessianStructure,Symbol,String}=:full::(_ in (":full", ":diagonal"))
    backend::Symbol=:GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64=1.0 
    μ₀::Float64=0.0 
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing
    link_approx::Symbol=:probit::(_ in (:probit,:plugin)) 
    predict_proba::Bool= true::(_ in (true,false)) 
    fit_prior_nsteps::Int=100::(_ > 0)
    
end
###############
const MLJ_Laplace= Union{LaplaceClassification,LaplaceRegression}

################################ functions shape and build 

function MLJFlux.shape(model::LaplaceClassification, X, y)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X
    n_input =  size(X,2)
    levels = unique(y)
    n_output = length(levels)
    return (n_input, n_output)
    
end

function MLJFlux.shape(model::LaplaceRegression, X, y)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X
    n_input =  size(X,2)
    dims = size(y)
        if length(dims) == 1
            n_output= 1
        else
            n_output= dims[1]
        end
        return (n_input, n_output)
end


function MLJFlux.build(model::LaplaceClassification,rng, shape)
    #chain
    chain = Flux.Chain(MLJFlux.build(model.builder, rng,shape...), model.finaliser)
    
    return chain
end


function MLJFlux.build(model::LaplaceRegression,rng,shape)
    #chain
    chain = MLJFlux.build(model.builder,rng , shape...)
    
    return chain
end

function MLJFlux.fitresult(model::LaplaceClassification, chain, y)
    return (chain,  length(unique(y)))
end

function MLJFlux.fitresult(model::LaplaceRegression, chain, y)
    return (chain, size(y)  )
end






#########################################  fit and predict for classification


"""
    MLJFlux.fit!(model::LaplaceClassification, chain,penalty,optimiser,epochs, verbosity, X, y)

Fit the LaplaceClassification model using MLJFlux.

# Arguments
- `model::LaplaceClassification`: The LaplaceClassification object to fit.
- `chain`: The chain to use during training.
- `penalty`: a penalty function to add to the loss function during training.
- `optimiser`: the optimiser to use during training.
- `epochs`: the number of epochs use for training.
- `verbosity`: The verbosity level for training.
- `X`: The input data for training.
- `y`: The target labels for training.

# Returns
- `model::LaplaceClassification`: The fitted LaplaceClassification model.
"""
function MLJFlux.fit!(model::LaplaceClassification, chain,penalty,optimiser,epochs, verbosity, X, y)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X
    X = X'

    # Integer encode the target variable y
    #y_onehot = unique(y) .== permutedims(y)

    la = LaplaceRedux.Laplace(
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
   loader = Flux.Data.DataLoader((data=X, label=y), batchsize=model.batch_size, shuffle=true)
   parameters = Flux.params(chain)
   for i in 1:epochs
       epoch_loss = 0.0
       # train the model
       for (X_batch, y_batch) in loader

           # Backward pass
           gs = Flux.gradient(parameters) do
            batch_loss =  (model.loss(chain(X_batch), y_batch) + penalty(Flux.params(chain))  )
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
   LaplaceRedux.fit!(la,zip(eachcol(X),y))
   optimize_prior!(la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)
   report=[]

   return (la, history, report)
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



"""
    MLJFlux.fit!(model::LaplaceRegression, penalty, chain, epochs, batch_size, optimiser, verbosity, X, y)

Fit the LaplaceRegression model using Flux.jl.

# Arguments
- `model::LaplaceRegression`: The LaplaceRegression model.
- `chain`: The chain to use during training.
- `penalty`: a penalty function to add to the loss function during training.
- `optimiser`: the optimiser to use during training.
- `epochs`: the number of epochs use for training.
- `verbosity`: The verbosity level for training.
- `X`: The input data for training.
- `y`: The target labels for training.

# Returns
- `model::LaplaceRegression`: The fitted LaplaceRegression model.
"""
function MLJFlux.fit!(model::LaplaceRegression, chain,penalty,optimiser,epochs, verbosity, X, y)
    X = X isa Tables.MatrixTable ? MLJBase.matrix(X) : X
    X = X'
    la = LaplaceRedux.Laplace(
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
    loader = Flux.Data.DataLoader((data=X, label=y), batchsize=model.batch_size, shuffle=true)
    parameters = Flux.params(chain)
    for i in 1:epochs
        epoch_loss = 0.0
        # train the model
        for (X_batch, y_batch) in loader
            y_batch = reshape(y_batch,1,:)
        
            # Backward pass
            gs = Flux.gradient(parameters) do
                batch_loss =  (model.loss(chain(X_batch), y_batch) + penalty(Flux.params(chain))  )
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
    LaplaceRedux.fit!(la,zip(eachcol(X),y))
    optimize_prior!(la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)
    report=[]

    return (la, history, report)
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

    #predictions = []
    #for row in eachrow(yhat)

        #mean_val = Float64(row[1][1][1])
        #std_val = sqrt(Float64(row[1][2][1]))
        # Append a Normal distribution:
        #push!(predictions, Normal(mean_val, std_val))
    #end
    
    return yhat

end



# Then for each model,
MLJBase.metadata_model(
    LaplaceClassification;
    input=Union{        
        AbstractMatrix{MLJBase.Finite},
        MLJBase.Table(MLJBase.Finite),
        AbstractMatrix{MLJBase.Continuous},
        MLJBase.Table(MLJBase.Continuous),
        MLJBase.Table{AbstractVector{MLJBase.Continuous}},
        MLJBase.Table{AbstractVector{MLJBase.Finite}}
    },
    target=Union{
        AbstractArray{MLJBase.Finite},
        AbstractArray{MLJBase.Continuous},
    },
    path="MLJFlux.LaplaceClassification",
)
# Then for each model,
MLJBase.metadata_model(
    LaplaceRegression;
    input=Union{
        AbstractMatrix{MLJBase.Continuous},
        MLJBase.Table(MLJBase.Continuous),
        AbstractMatrix{MLJBase.Finite},
        MLJBase.Table(MLJBase.Finite),
        MLJBase.Table{AbstractVector{MLJBase.Continuous}},
        MLJBase.Table{AbstractVector{MLJBase.Finite}}
    },
    target=Union{
        AbstractArray{MLJBase.Finite},
        AbstractArray{MLJBase.Continuous},
    },
    path="MLJFlux.LaplaceRegression",
)
 