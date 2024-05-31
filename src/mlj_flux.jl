using Flux
using MLJFlux
import MLJModelInterface as MMI
using ProgressMeter: Progress, next!, BarGlyphs
using Random
using Tables
using ComputationalResources
using Statistics
using Distributions
using LinearAlgebra

mutable struct LaplaceClassification{B,F,O,L} <: MLJFlux.MLJFluxProbabilistic
    builder::B
    finaliser::F
    optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L        # can be called as in `loss(yhat, y)`
    epochs::Int    # number of epochs
    batch_size::Int  # size of a batch
    lambda::Float64  # regularization strength
    alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
    rng::Union{AbstractRNG,Int64}
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
    likelihood::Symbol
    subset_of_weights::Symbol
    subnetwork_indices::Vector{Vector{Int}}
    hessian_structure::Union{HessianStructure,Symbol,String}
    backend::Symbol
    σ::Real
    μ₀::Real
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}
    link_approx::Symbol
    fit_params::Dict{Symbol,Any}
    la::Union{Nothing,AbstractLaplace}
end

"""
    LaplaceApproximation(; builder, finaliser, optimiser, loss, epochs, batch_size, lambda, alpha, rng, optimiser_changes_trigger_retraining, acceleration, likelihood, subset_of_weights, subnetwork_indices, hessian_structure, backend, σ, μ₀, P₀, link_approx, fit_params)

A probabilistic model that uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. The model is trained using the `fit!` method. The model is defined by the following default parameters for all `MLJFlux` models:

- `builder`: a Flux model that constructs the neural network.
- `finaliser`: a Flux model that processes the output of the neural network.
- `optimiser`: a Flux optimiser.
- `loss`: a loss function that takes the predicted output and the true output as arguments.
- `epochs`: the number of epochs to train the model.
- `batch_size`: the size of a batch.
- `lambda`: the regularization strength.
- `alpha`: the regularization mix (0 for all l2, 1 for all l1).
- `rng`: a random number generator.
- `optimiser_changes_trigger_retraining`: a boolean indicating whether changes in the optimiser trigger retraining.
- `acceleration`: the computational resource to use.

The model also has the following parameters, which are specific to the Laplace approximation:

- `likelihood`: the likelihood of the model, either `:classification` or `:regression`.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `link_approx`: the link approximation to use, either `:probit` or `:plugin`.
- `fit_params`: additional parameters to pass to the `fit!` method.
"""
function LaplaceClassification(;
    builder::B=MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish),
    finaliser::F=Flux.softmax,
    optimiser::O=Flux.Optimise.Adam(),
    loss::L=Flux.crossentropy,
    epochs::Int=10,
    batch_size::Int=1,
    lambda::Float64=1.0,
    alpha::Float64=0.0,
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG,
    optimiser_changes_trigger_retraining::Bool=false,
    acceleration::AbstractResource=CPU1(),
    subset_of_weights::Symbol=:all,
    subnetwork_indices::Vector{Vector{Int}}=Vector{Vector{Int}}([]),
    hessian_structure::Union{HessianStructure,Symbol,String}=:full,
    backend::Symbol=:GGN,
    σ::Float64=1.0,
    μ₀::Float64=0.0,
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing,
    link_approx::Symbol=:probit,
    fit_params::Dict{Symbol,Any}=Dict{Symbol,Any}(:override => true),
) where {B,F,O,L}
    likelihood = :classification
    model = LaplaceClassification(
        builder,
        finaliser,
        optimiser,
        loss,
        epochs,
        batch_size,
        lambda,
        alpha,
        rng,
        optimiser_changes_trigger_retraining,
        acceleration,
        likelihood,
        subset_of_weights,
        subnetwork_indices,
        hessian_structure,
        backend,
        σ,
        μ₀,
        P₀,
        link_approx,
        fit_params,
        nothing,
    )

    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end

mutable struct LaplaceRegression{B,F,O,L} <: MLJFlux.MLJFluxProbabilistic
    builder::B
    finaliser::F
    optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L        # can be called as in `loss(yhat, y)`
    epochs::Int    # number of epochs
    batch_size::Int  # size of a batch
    lambda::Float64  # regularization strength
    alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
    rng::Union{AbstractRNG,Int64}
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
    likelihood::Symbol
    subset_of_weights::Symbol
    subnetwork_indices::Vector{Vector{Int}}
    hessian_structure::Union{HessianStructure,Symbol,String}
    backend::Symbol
    σ::Real
    μ₀::Real
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}
    fit_params::Dict{Symbol,Any}
    la::Union{Nothing,AbstractLaplace}
end

"""
    LaplaceRegression(; builder, finaliser, optimiser, loss, epochs, batch_size, lambda, alpha, rng, optimiser_changes_trigger_retraining, acceleration, likelihood, subset_of_weights, subnetwork_indices, hessian_structure, backend, σ, μ₀, P₀, link_approx, fit_params)

A probabilistic regression model that uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. The model is trained using the `fit!` method. The model is defined by the following default parameters for all `MLJFlux` models:

- `builder`: a Flux model that constructs the neural network.
- `finaliser`: a Flux model that processes the output of the neural network.
- `optimiser`: a Flux optimiser.
- `loss`: a loss function that takes the predicted output and the true output as arguments.
- `epochs`: the number of epochs to train the model.
- `batch_size`: the size of a batch.
- `lambda`: the regularization strength.
- `alpha`: the regularization mix (0 for all l2, 1 for all l1).
- `rng`: a random number generator.
- `optimiser_changes_trigger_retraining`: a boolean indicating whether changes in the optimiser trigger retraining.
- `acceleration`: the computational resource to use.

The model also has the following parameters, which are specific to the Laplace approximation:

- `likelihood`: the likelihood of the model `:regression`.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `fit_params`: additional parameters to pass to the `fit!` method.
"""
function LaplaceRegression(;
    builder::B=MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish),
    finaliser::F=x->x,
    optimiser::O=Flux.Optimise.Adam(),
    loss::L=Flux.mse,
    epochs::Int=10,
    batch_size::Int=1,
    lambda::Float64=1.0,
    alpha::Float64=0.0,
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG,
    optimiser_changes_trigger_retraining::Bool=false,
    acceleration::AbstractResource=CPU1(),
    subset_of_weights::Symbol=:all,
    subnetwork_indices::Vector{Vector{Int}}=Vector{Vector{Int}}([]),
    hessian_structure::Union{HessianStructure,Symbol,String}=:full,
    backend::Symbol=:GGN,
    σ::Float64=1.0,
    μ₀::Float64=0.0,
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing,
    fit_params::Dict{Symbol,Any}=Dict{Symbol,Any}(:override => true),
) where {B,F,O,L}
    likelihood=:regression
    model = LaplaceRegression(
        builder,
        finaliser,
        optimiser,
        loss,
        epochs,
        batch_size,
        lambda,
        alpha,
        rng,
        optimiser_changes_trigger_retraining,
        acceleration,
        likelihood,
        subset_of_weights,
        subnetwork_indices,
        hessian_structure,
        backend,
        σ,
        μ₀,
        P₀,
        fit_params,
        nothing,
    )

    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end







function MLJFlux.shape(model::Union{LaplaceClassification,LaplaceRegression}, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    n_input =  length(Tables.columnnames(X))

    if model isa LaplaceClassification
        levels = MMI.classes(y[1])
        n_output = length(levels)
        return (n_input, n_output)
    elseif model isa LaplaceRegression
        dims = size(y)
        if length(dims) == 1
            n_output= 1
        else
            n_output= dims[2]
        end
        return (n_input, n_output)
    end

end


function MLJFlux.build(model::Union{LaplaceClassification,LaplaceRegression}, rng, shape)
    # Construct the chain
    chain = Flux.Chain(MLJFlux.build(model.builder, rng, shape...), model.finaliser)
    # Construct Laplace model and store it in the model object
    model.la = Laplace(
        chain;
        likelihood=model.likelihood,
        subset_of_weights=model.subset_of_weights,
        subnetwork_indices=model.subnetwork_indices,
        hessian_structure=model.hessian_structure,
        backend=model.backend,
        σ=model.σ,
        μ₀=model.μ₀,
        P₀=model.P₀,
    )
    return chain
end

function MLJFlux.fitresult(model::Union{LaplaceClassification,LaplaceRegression}, chain, y)
    if model isa LaplaceClassification
        return (chain, model.la, MMI.classes(y[1]))
    else
        return (chain, model.la, size(y)  )
    end
end


function MMI.clean!(model::Union{LaplaceClassification,LaplaceRegression})
    warning = ""
    if model.lambda < 0
        warning *= "Need `lambda ≥ 0`. Resetting `lambda = 0`. "
        model.lambda = 0
    end
    if model.alpha < 0 || model.alpha > 1
        warning *= "Need alpha in the interval `[0, 1]`. " * "Resetting `alpha = 0`. "
        model.alpha = 0
    end
    if model.epochs < 0
        warning *= "Need `epochs ≥ 0`. Resetting `epochs = 10`. "
        model.epochs = 10
    end
    if model.batch_size <= 0
        warning *= "Need `batch_size > 0`. Resetting `batch_size = 1`. "
        model.batch_size = 1
    end
    if model.acceleration isa CUDALibs && gpu_isdead()
        warning *=
            "`acceleration isa CUDALibs` " * "but no CUDA device (GPU) currently live. "
    end
    if !(model.acceleration isa CUDALibs || model.acceleration isa CPU1)
        warning *= "`Undefined acceleration, falling back to CPU`"
        model.acceleration = CPU1()
    end
    if model.likelihood ∉ (:regression, :classification)
        warning *= "Need `likelihood ∈ (:regression, :classification)`. " *
        "Resetting to default `likelihood = :regression`. "
        model.likelihood = :regression
    end
    if model.subset_of_weights ∉ (:all, :last_layer, :subnetwork)
        warning *=
            "Need `subset_of_weights ∈ (:all, :last_layer, :subnetwork)`. " *
            "Resetting `subset_of_weights = :all`. "
        model.subset_of_weights = :all
    end
    if String(model.hessian_structure) ∉ ("full", "diagonal") &&
        !(typeof(model.hessian_structure) <: HessianStructure)
        warning *=
            "Need `hessian_structure ∈ (:full, :diagonal)` or `hessian_structure ∈ (:full, :diagonal)` or typeof(model.hessian_structure) <: HessianStructure." *
            "Resetting `hessian_structure = :full`. "
        model.hessian_structure = :full
    end
    if model.backend ∉ (:GGN, :EmpiricalFisher)
        warning *=
            "Need `backend ∈ (:GGN, :EmpiricalFisher)`. " * "Resetting `backend = :GGN`. "
        model.backend = :GGN
    end
    if model.likelihood == :classification &&  model.link_approx ∉ (:probit, :plugin)
        warning *=
            "Need `link_approx ∈ (:probit, :plugin)`. " *
            "Resetting `link_approx = :probit`. "
        model.link_approx = :probit
    end
    return warning
end
######################################### train , fit and predict for classification

function MLJFlux.train!(model::LaplaceClassification, penalty, chain, optimiser, X, y)
    loss = model.loss
    n_batches = length(y)
    training_loss = zero(Float32)
    for i in 1:n_batches
        parameters = Flux.params(chain)
        gs = Flux.gradient(parameters) do
            yhat = chain(X[i])
            batch_loss = loss(yhat, y[i]) + penalty(parameters) / n_batches
            training_loss += batch_loss
            return batch_loss
        end
        Flux.update!(optimiser, parameters, gs)
    end
    return training_loss / n_batches
end

function MLJFlux.fit!(
    model::LaplaceClassification, penalty, chain, optimiser, epochs, verbosity, X, y
)
    loss = model.loss

    # intitialize and start progress meter:
    meter = Progress(
        epochs + 1;
        dt=0,
        desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=25,
        color=:yellow,
    )
    verbosity != 1 || next!(meter)

    # initiate history:
    n_batches = length(y)

    parameters = Flux.params(chain)

    # initial loss:
    losses = (
        loss(chain(X[i]), y[i]) + penalty(parameters) / n_batches for i in 1:n_batches
    )
    history = [mean(losses)]

    for i in 1:epochs
        current_loss = MLJFlux.train!(
            model::MLJFlux.MLJFluxModel, penalty, chain, optimiser, X, y
        )
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    la = model.la

    # fit the Laplace model:
    fit!(la, zip(X, y); model.fit_params...)
    optimize_prior!(la; verbose=false, n_steps=100)

    model.la = la

    return chain, history
end



function MMI.predict(model::LaplaceClassification, fitresult, Xnew)
    chain, la, levels = fitresult
    # re-format Xnew into acceptable input for Laplace:
    X = MLJFlux.reformat(Xnew)
    # predict using Laplace:
    yhat = vcat(
        [
            predict(la, MLJFlux.tomat(X[:, i]); link_approx=model.link_approx)' for
            i in 1:size(X, 2)
        ]...,
    )

    return MMI.UnivariateFinite(levels, yhat)


end


######################################################## train ,fit and predict for regression




function MLJFlux.train!(model::LaplaceRegression, penalty, chain, optimiser, X, y)
    loss = model.loss
    n_batches = length(y)
    training_loss = zero(Float32)
    for i in 1:n_batches
        parameters = Flux.params(chain)
        gs = Flux.gradient(parameters) do
            yhat = chain(X[i])
            batch_loss = loss(yhat, y[i]) + penalty(parameters) / n_batches
            training_loss += batch_loss
            return batch_loss
        end
        Flux.update!(optimiser, parameters, gs)
    end
    return training_loss / n_batches
end

function MLJFlux.fit!(
    model::LaplaceRegression, penalty, chain, optimiser, epochs, verbosity, X, y
)
    loss = model.loss

    # intitialize and start progress meter:
    meter = Progress(
        epochs + 1;
        dt=0,
        desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=25,
        color=:yellow,
    )
    verbosity != 1 || next!(meter)

    # initiate history:
    n_batches = length(y)

    parameters = Flux.params(chain)

    # initial loss:
    losses = (
        loss(chain(X[i]), y[i]) + penalty(parameters) / n_batches for i in 1:n_batches
    )
    history = [mean(losses)]

    for i in 1:epochs
        current_loss = MLJFlux.train!(
            model::MLJFlux.MLJFluxModel, penalty, chain, optimiser, X, y
        )
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    la = model.la

    # fit the Laplace model:
    fit!(la, zip(X, y); model.fit_params...)
    optimize_prior!(la; verbose=false, n_steps=100)

    model.la = la

    return chain, history
end


function MMI.predict(model::LaplaceRegression, fitresult, Xnew)
    chain, la, levels = fitresult
    # re-format Xnew into acceptable input for Laplace:
    X = MLJFlux.reformat(Xnew)
    # predict using Laplace:
    yhat = vcat(
        [
            glm_predictive_distribution(la, MLJFlux.tomat(X[:, i]))' for
            i in 1:size(X, 2)
        ]...,
    )
    println(size(yhat))
    predictions = []
    for row in eachrow(yhat)
 
        mean_val = Float64(row[1][1])
        std_val = sqrt(Float64(row[2][1]))
        # Append a Normal distribution:
        push!(predictions, Normal(mean_val, std_val))
    end
    
    return predictions

end
