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
- flux_model????
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
    flux_model::Flux 
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    ret_distr::Bool = false::(_ in (true, false))
    fit_prior_nsteps::Int = 100::(_ > 0)
end


function MMI.fit(m::LaplaceRegressor, verbosity, X, y, w=nothing)
    features  = Tables.schema(X).names

    X = MLJBase.matrix(X)

    la = LaplaceRedux.Laplace(
        m.flux_model;
        likelihood=:regression,
        subset_of_weights=model.subset_of_weights,
        subnetwork_indices=model.subnetwork_indices,
        hessian_structure=model.hessian_structure,
        backend=model.backend,
        σ=m.σ,
        μ₀=m.μ₀,
        P₀=m.P₀,
    )

    # fit the Laplace model:
    LaplaceRedux.fit!(la, zip(X, y))
    optimize_prior!(la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)

    
    fitresult=la
    report = (status="success", message="Model fitted successfully")
    cache     = nothing
    return (fitresult, cache, report)
end




function MMI.predict(m::LaplaceRegressor, fitresult, Xnew)
    Xnew = MLJBase.matrix(Xnew) |> permutedims
    la = fitresult[1]
    yhat = LaplaceRedux.predict(la, Xnew; ret_distr=model.ret_distr)
    return [Normal(μᵢ, σ̂) for (μᵢ,σ) ∈ yhat]
end





"""
    MLJBase.@mlj_model mutable struct LaplaceClassifier <: MLJFlux.MLJFluxProbabilistic

A mutable struct representing a Laplace Classification model.
It uses Laplace approximation to estimate the posterior distribution of the weights of a neural network. 


The model also has the following parameters:

- `model`: Flux ???
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

    model::Flux 
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
    ret_distr::Bool = false::(_ in (true, false))
    fit_prior_nsteps::Int = 100::(_ > 0)
end



function MMI.fit(m::LaplaceClassifier, verbosity, X, y, w=nothing)
    features  = Tables.schema(X).names
    Xmatrix = MLJBase.matrix(X)
    decode    = y[1]
    y_plain   = MLJBase.int(y) .- 1 # 0, 1 of type Int

    la = LaplaceRedux.Laplace(
        m.flux_model;
        likelihood=:classification,
        subset_of_weights=model.subset_of_weights,
        subnetwork_indices=model.subnetwork_indices,
        hessian_structure=model.hessian_structure,
        backend=model.backend,
        σ=m.σ,
        μ₀=m.μ₀,
        P₀=m.P₀,
    )

    # fit the Laplace model:
    LaplaceRedux.fit!(la, zip(X, y_plain))
    optimize_prior!(la; verbose=verbose_laplace, n_steps=model.fit_prior_nsteps)

    
    fitresult=la
    report = (status="success", message="Model fitted successfully")
    cache     = nothing
    return (fitresult, decode), cache, report
end


function MMI.predict(m::LaplaceClassifier, (fitresult, decode), Xnew)
    la = fitresult
    Xnew = MLJBase.matrix(Xnew) |> permutedims
    predictions = LaplaceRedux.predict(
        la,
        Xnew;
        link_approx=model.link_approx,
        predict_proba=model.predict_proba,
        ret_distr=model.ret_distr,
    )
    return [MLJBase.UnivariateFinite(MLJBase.classes(decode), prediction) for prediction in predictions]
end




MLJBase.metadata_model(
    LaplaceClassifier;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite, MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Contintuous), # table with mixed types
    },
    target_scitype=AbstractArray{<:MLJBase.Finite}, # ordered factor or multiclass
    load_path="LaplaceRedux.LaplaceClassification",
)
# metadata for each model,
MLJBase.metadata_model(
    LaplaceRegressor;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite, MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Contintuous), # table with mixed types
    },
     target_scitype=AbstractArray{MLJBase.Continuous},
    load_path="LaplaceRedux.MLJFlux.LaplaceRegression",
)