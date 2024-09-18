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
- `flux_model`: A flux model provided by the user and compatible with the dataset.
- `epochs`: The number of training epochs.
- `batch_size`: The batch size.
- `subset_of_weights`: the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.
- `subnetwork_indices`: the indices of the subnetworks.
- `hessian_structure`: the structure of the Hessian matrix, either `:full` or `:diagonal`.
- `backend`: the backend to use, either `:GGN` or `:EmpiricalFisher`.
- `σ`: the standard deviation of the prior distribution.
- `μ₀`: the mean of the prior distribution.
- `P₀`: the covariance matrix of the prior distribution.
- `ret_distr`: a boolean that tells predict to either return distributions (true) objects from Distributions.jl or just the probabilities.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
"""
MLJBase.@mlj_model mutable struct LaplaceRegressor <: MLJFlux.MLJFluxProbabilistic

    flux_model::Flux.Chain = nothing
    flux_loss = Flux.Losses.mse
    epochs::Integer = 1000::(_ > 0)
    batch_size::Integer= 32::(_ > 0)
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    #ret_distr::Bool = false::(_ in (true, false))
    fit_prior_nsteps::Int = 100::(_ > 0)
end


function MMI.fit(m::LaplaceRegressor, verbosity, X, y)
    #features  = Tables.schema(X).names

    X = MLJBase.matrix(X) |> permutedims
    y = reshape(y, 1,:)
    data_loader = Flux.DataLoader((X,y), batchsize=m.batch_size)
    opt_state = Flux.setup(Adam(), m.flux_model)

    for epoch in 1:m.epochs
        Flux.train!(m.flux_model,data_loader, opt_state) do model, X, y
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
    LaplaceRedux.fit!(la, data_loader )
    optimize_prior!(la; verbose= false, n_steps=m.fit_prior_nsteps)

    
    fitresult=la
    report = (status="success", message="Model fitted successfully")
    cache     = nothing
    return (fitresult, cache, report)
end




function MMI.predict(m::LaplaceRegressor, fitresult, Xnew)
    Xnew = MLJBase.matrix(Xnew) |> permutedims
    la = fitresult
    yhat = LaplaceRedux.predict(la, Xnew; ret_distr= false)
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

- `flux_model`: A flux model provided by the user and compatible with the dataset.
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
- `predict_proba`: a boolean that select whether to predict probabilities or not.
- `ret_distr`: a boolean that tells predict to either return distributions (true) objects from Distributions.jl or just the probabilities.
- `fit_prior_nsteps`: the number of steps used to fit the priors.
"""
MLJBase.@mlj_model mutable struct LaplaceClassifier <: MLJFlux.MLJFluxProbabilistic

    flux_model::Flux.Chain = nothing
    flux_loss = Flux.Losses.logitcrossentropy
    epochs::Integer = 1000::(_ > 0)
    batch_size::Integer= 32::(_ > 0)
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    #ret_distr::Bool = false::(_ in (true, false))
    fit_prior_nsteps::Int = 100::(_ > 0)
    link_approx::Symbol = :probit::(_ in (:probit, :plugin))
end



function MMI.fit(m::LaplaceClassifier, verbosity, X, y)
    #features  = Tables.schema(X).names
    X = MLJBase.matrix(X) |> permutedims
    decode = MMI.decoder(y[1])
    #y = reshape(y, 1,:)
    y_plain   = MLJBase.int(y) .- 1 # 0, 1 of type Int
    y_onehot = Flux.onehotbatch(y_plain,  unique(y_plain) )



    #loss(y_hat, y) =  Flux.Losses.logitcrossentropy(y_hat, y)

    data_loader = Flux.DataLoader((X,y_onehot), batchsize=m.batch_size)
    opt_state = Flux.setup(Adam(), m.flux_model)



    for epoch in 1:m.epochs
        Flux.train!(m.flux_model,data_loader, opt_state) do model, X, y_onehot
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
    LaplaceRedux.fit!(la, data_loader )
    optimize_prior!(la; verbose= false, n_steps=m.fit_prior_nsteps)

    report = (status="success", message="Model fitted successfully")
    cache     = nothing
    return ((la,decode), cache, report)
end


function MMI.predict(m::LaplaceClassifier, (fitresult, decode), Xnew)
    la = fitresult
    Xnew = MLJBase.matrix(Xnew) |> permutedims
    predictions = LaplaceRedux.predict(
        la,
        Xnew;
        link_approx=m.link_approx,
        ret_distr=false)
    return [MLJBase.UnivariateFinite(MLJBase.classes(decode), prediction, pool=decode,augment=true) for prediction in predictions]
end




MLJBase.metadata_model(
    LaplaceClassifier;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite, MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Continuous), # table with mixed types
    },
    target_scitype=AbstractArray{<:MLJBase.Finite}, # ordered factor or multiclass
    load_path="LaplaceRedux.LaplaceClassifier",
)
# metadata for each model,
MLJBase.metadata_model(
    LaplaceRegressor;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite, MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Continuous), # table with mixed types
    },
     target_scitype=AbstractArray{MLJBase.Continuous},
    load_path="LaplaceRedux.LaplaceRegressor",
)