using .Curvature: Kron, KronDecomposed, mm
using LinearAlgebra
using MLUtils

"Abstract base type for all Laplace approximations in this library. All subclasses implemented are parametric."
abstract type AbstractLaplace end

"""
    LaplaceParams

Container for the parameters of a Laplace approximation.

# Fields

- `subset_of_weights::Symbol`: the subset of weights to consider. Possible values are `:all`, `:last_layer`, and `:subnetwork`.
- `subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}`: the indices of the subnetwork. Possible values are `nothing` or a vector of vectors of integers.
- `hessian_structure::Symbol`: the structure of the Hessian. Possible values are `:full` and `:kron`.
- `backend::Symbol`: the backend to use. Possible values are `:GGN` and `:Fisher`.
- `curvature::Union{Curvature.CurvatureInterface,Nothing}`: the curvature interface. Possible values are `nothing` or a concrete subtype of `CurvatureInterface`.
- `σ::Real`: the observation noise
- `μ₀::Real`: the prior mean
- `λ::Real`: the prior precision
- `P₀::Union{Nothing,AbstractMatrix,UniformScaling}`: the prior precision matrix
"""
Base.@kwdef struct LaplaceParams
    subset_of_weights::Symbol = :all
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}} = nothing
    hessian_structure::Symbol = :full
    backend::Symbol = :GGN
    curvature::Union{Curvature.CurvatureInterface,Nothing} = nothing
    σ::Real = 1.0
    μ₀::Real = 0.0
    λ::Real = 1.0
    P₀::Union{Nothing,AbstractMatrix,UniformScaling} = nothing
end

"""
    EstimationParams

Container for the parameters of a Laplace approximation.
"""
mutable struct EstimationParams
    subset_of_weights::Symbol 
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}} 
    hessian_structure::Symbol 
    curvature::Union{Curvature.CurvatureInterface,Nothing} 
end

"""
    EstimationParams(params::LaplaceParams)

Extracts the estimation parameters from a `LaplaceParams` object.
"""
function EstimationParams(params::LaplaceParams, model::Any, likelihood::Symbol)

    # Asserts:
    @assert params.subset_of_weights ∈ [:all, :last_layer, :subnetwork] "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer, :subnetwork]`"
    if (params.subset_of_weights == :subnetwork)
        validate_subnetwork_indices(params.subnetwork_indices, Flux.params(model))
    end

    est_params = EstimationParams(
        params.subset_of_weights,
        params.subnetwork_indices,
        params.hessian_structure,
        params.curvature,
    )

    # Instantiating curvature interface
    instantiate_curvature!(est_params, model, likelihood, params.backend)

    return est_params
end

"""
    instantiate_curvature!(params::EstimationParams, model::Any, likelihood::Symbol, backend::Symbol)

Instantiates the curvature interface for a Laplace approximation. The curvature interface is a concrete subtype of [`CurvatureInterface`](@ref) and is used to compute the Hessian matrix. The curvature interface is stored in the `curvature` field of the `EstimationParams` object.
"""
function instantiate_curvature!(params::EstimationParams, model::Any, likelihood::Symbol, backend::Symbol)

    model_params = Flux.params(model)
    n_elements = length(model_params)
    subnetwork_indices = nothing
    if params.subset_of_weights == :all || la.subset_of_weights == :subnetwork
        # get all parameters and constants in logitbinarycrossentropy
        model_params = [θ for θ in model_params]
    elseif params.subset_of_weights == :last_layer
        # Only get last layer parameters:
        # params[n_elements] is the bias vector of the last layer
        # params[n_elements-1] is the weight matrix of the last layer
        model_params = [model_params[n_elements - 1], model_params[n_elements]]
    elseif params.subset_of_weights == :subnetwork
        subnetwork_indices = convert_subnetwork_indices(params.subnetwork_indices, model_params)
    end

    curvature = getfield(Curvature, backend)(
        model,
        likelihood,
        model_params,
        params.subset_of_weights,
        subnetwork_indices,
    )

    params.curvature = curvature
end

"""
    Prior

Container for the prior parameters of a Laplace approximation.
"""
mutable struct Prior
    σ::Real 
    μ₀::Real 
    λ::Real 
    P₀::Union{Nothing,AbstractMatrix,UniformScaling}
end

"""
    Prior(params::LaplaceParams)

Extracts the prior parameters from a `LaplaceParams` object.
"""
Prior(params::LaplaceParams) = Prior(params.σ, params.μ₀, params.λ, params.P₀)

"""
    Laplace

Concrete type for Laplace approximation. This type is a subtype of `AbstractLaplace` and is used to store all the necessary information for a Laplace approximation.

# Fields

- `model::Flux.Chain`: The model to be approximated.
- `likelihood::Symbol`: The likelihood function to be used.
- `prior::Prior`: The parameters defining prior distribution.
- `params::EstimationParams`: The estimation parameters.
"""
mutable struct Laplace <: AbstractLaplace
    model::Flux.Chain
    likelihood::Symbol
    prior::Prior
    params::EstimationParams
end

"""
Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)

Outer constructor for Laplace approximation. This function is a wrapper around the [`EstimationParams`](@ref) constructor and the [`Laplace`](@ref) constructor.
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...)

    # Load hyperparameters:
    args = LaplaceParams(; kwargs...)
    @assert !(args.σ != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."

    # Unpack arguments and wrap in containers:
    est_args = EstimationParams(args, model, likelihood)
    prior = Prior(args)

    # Instantiate Laplace object:
    la = Laplace(model, likelihood, prior, est_args)

    la.μ = la.μ[(end - la.n_params + 1):end]                                    # adjust weight vector
    if typeof(la.P₀) <: UniformScaling
        la.P₀ = la.P₀(la.n_params)
    end

    # Sanity:
    if isa(la.P₀, AbstractMatrix)
        @assert all(size(la.P₀) .== la.n_params) "Dimensions of prior Hessian $(size(la.P₀)) do not align with number of parameters ($(la.n_params))"
    end

    return la
end

"""
    Fitresult

Container for the results of a Laplace approximation.

# Fields

- `la::AbstractLaplace`: the Laplace approximation object
- `μ::AbstractVector`: the MAP estimate of the parameters
- `H::Union{AbstractArray,KronDecomposed,Nothing}`: the Hessian matrix
- `P::Union{AbstractArray,KronDecomposed,Nothing}`: the posterior precision matrix
- `Σ::Union{AbstractArray,Nothing}`: the posterior covariance matrix
- `n_params::Union{Int,Nothing}`: the number of parameters
- `n_data::Union{Int,Nothing}`: the number of data points
- `n_out::Union{Int,Nothing}`: the number of outputs
- `loss::Real`: the loss value
"""
mutable struct Fitresult
    la::AbstractLaplace
    μ::AbstractVector
    H::Union{AbstractArray,KronDecomposed,Nothing}
    P::Union{AbstractArray,KronDecomposed,Nothing}
    Σ::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
    n_data::Union{Int,Nothing}
    n_out::Union{Int,Nothing}
    loss::Real 
end

"""
    outdim(la::AbstractLaplace)

Helper function to determine the output dimension, corresponding to the number of neurons 
on the last layer of the NN, of a `Flux.Chain` with Laplace approximation.
"""
outdim(la::AbstractLaplace) = outdim(la.model)

@doc raw"""
    posterior_precision(la::AbstractLaplace)

Computes the posterior precision ``P`` for a fitted Laplace Approximation as follows,

``
P = \sum_{n=1}^N\nabla_{\theta}^2\log p(\mathcal{D}_n|\theta)|_{\theta}_{MAP} + \nabla_{\theta}^2 \log p(\theta)|_{\theta}_{MAP} 
``

where ``\sum_{n=1}^N\nabla_{\theta}^2\log p(\mathcal{D}_n|\theta)|_{\theta}_{MAP}=H`` and ``\nabla_{\theta}^2 \log p(\theta)|_{\theta}_{MAP}=P_0``.
"""
function posterior_precision(la::AbstractLaplace, H=la.H, P₀=la.P₀)
    @assert !isnothing(H) "Hessian not available. Either no value supplied or Laplace Approximation has not yet been estimated."
    return H + P₀
end

@doc raw"""
    posterior_covariance(la::AbstractLaplace, P=la.P)

Computes the posterior covariance ``∑`` as the inverse of the posterior precision: ``\Sigma=P^{-1}``.
"""
function posterior_covariance(la::AbstractLaplace, P=posterior_precision(la))
    @assert !isnothing(P) "Posterior precision not available. Either no value supplied or Laplace Approximation has not yet been estimated."
    return inv(P)
end

"""
    log_likelihood(la::AbstractLaplace)


"""
function log_likelihood(la::AbstractLaplace)
    factor = -_H_factor(la)
    if la.likelihood == :regression
        c = la.n_data * la.n_out * log(la.σ * sqrt(2 * pi))
    else
        c = 0
    end
    return factor * la.loss - c
end

"""
    _H_factor(la::AbstractLaplace)

Returns the factor σ⁻², where σ is used in the zero-centered Gaussian prior p(θ) = N(θ;0,σ²I)
"""
_H_factor(la::AbstractLaplace) = 1 / (la.σ^2)

"""
    _init_H(la::AbstractLaplace)


"""
_init_H(la::AbstractLaplace) = zeros(la.n_params, la.n_params)

"""
    _weight_penalty(la::AbstractLaplace)

The weight penalty term is a regularization term used to prevent overfitting.
Weight regularization methods such as weight decay introduce a penalty to the loss function when training a neural network to encourage the network to use small weights.
Smaller weights in a neural network can result in a model that is more stable and less likely to overfit the training dataset, in turn having better performance when 
making a prediction on new data.
"""
function _weight_penalty(la::AbstractLaplace)
    μ = la.μ                                                                 # MAP
    μ₀ = la.μ₀                                                               # prior
    Δ = μ .- μ₀
    return Δ'la.P₀ * Δ                                                       # measure of how far the MAP estimate deviates from the prior mean μ₀
end                                                                          # used to control the degree of regularization applied to the mode

"""
    log_marginal_likelihood(la::AbstractLaplace; P₀::Union{Nothing,UniformScaling}=nothing, σ::Union{Nothing, Real}=nothing)


"""
function log_marginal_likelihood(
    la::AbstractLaplace;
    P₀::Union{Nothing,AbstractFloat,AbstractMatrix}=nothing,
    σ::Union{Nothing,Real}=nothing,
)

    # update prior precision:
    if !isnothing(P₀)
        la.P₀ = typeof(P₀) <: AbstractFloat ? UniformScaling(P₀)(la.n_params) : P₀
    end

    # update observation noise:
    if !isnothing(σ)
        @assert (la.likelihood == :regression || la.σ == σ) "Can only change observational noise σ for regression."
        la.σ = σ
    end

    return log_likelihood(la) - 0.5 * (log_det_ratio(la) + _weight_penalty(la))
end

"""
    log_det_ratio(la::AbstractLaplace)


"""
function log_det_ratio(la::AbstractLaplace)
    return log_det_posterior_precision(la) - log_det_prior_precision(la)
end

"""
    log_det_prior_precision(la::AbstractLaplace)


"""
log_det_prior_precision(la::AbstractLaplace) = sum(log.(diag(la.P₀)))

"""
    log_det_posterior_precision(la::AbstractLaplace)


"""
log_det_posterior_precision(la::AbstractLaplace) = logdet(posterior_precision(la))

"""
    hessian_approximation(la::AbstractLaplace, d; batched::Bool=false)

Computes the local Hessian approximation at a single datapoint `d`.
"""
function hessian_approximation(la::AbstractLaplace, d; batched::Bool=false)
    loss, H = getfield(Curvature, la.hessian_structure)(la.curvature, d; batched=batched)
    return loss, H
end

"""
    fit!(la::AbstractLaplace,data)

Fits the Laplace approximation for a data set.
The function returns the number of observations (n_data) that were used to update the Laplace object.
It does not return the updated Laplace object itself because the function modifies the input Laplace object in place (as denoted by the use of '!' in the function's name).

# Examples

```julia-repl
using Flux, LaplaceRedux
x, y = LaplaceRedux.Data.toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn)
fit!(la, data)
```

"""
function fit!(la::AbstractLaplace, data; override::Bool=true)
    return _fit!(la, data; batched=false, batchsize=1, override=override)
end

"""
Fit the Laplace approximation, with batched data.
"""
function fit!(la::AbstractLaplace, data::DataLoader; override::Bool=true)
    return _fit!(la, data; batched=true, batchsize=data.batchsize, override=override)
end

"""
    glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.curvature, X)
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

# Posterior predictions:
"""
    predict(la::AbstractLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true)

Computes predictions from Bayesian neural network.

# Examples

```julia-repl
using Flux, LaplaceRedux
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn)
fit!(la, data)
predict(la, hcat(x...))
```

"""
function predict(
    la::AbstractLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true
)
    fμ, fvar = glm_predictive_distribution(la, X)

    # Regression:
    if la.likelihood == :regression
        return fμ, fvar
    end

    # Classification:
    if la.likelihood == :classification

        # Probit approximation
        if link_approx == :probit
            κ = 1 ./ sqrt.(1 .+ π / 8 .* fvar)
            z = κ .* fμ
        end

        if link_approx == :plugin
            z = fμ
        end

        # Sigmoid/Softmax
        if predict_proba
            if outdim(la) == 1
                p = Flux.sigmoid(z)
            else
                p = Flux.softmax(z; dims=1)
            end
        else
            p = z
        end

        return p
    end
end

"""
Compute predictive posteriors for a batch of inputs.

Note, input is assumed to be batched only if it is a matrix.
If the input dimensionality of the model is 1 (a vector), one should still prepare a 1×B matrix batch as input.
"""
function predict(la::AbstractLaplace, X::Matrix; link_approx=:probit, predict_proba::Bool=true)
    return stack([
        predict(la, X[:, i]; link_approx=link_approx, predict_proba=predict_proba) for
        i in 1:size(X, 2)
    ])
end

"""
    (la::AbstractLaplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::AbstractLaplace)(X::AbstractArray; kwrgs...)
    return predict(la, X; kwrgs...)
end

"""
    optimize_prior!(
        la::AbstractLaplace; 
        n_steps::Int=100, lr::Real=1e-1,
        λinit::Union{Nothing,Real}=nothing,
        σinit::Union{Nothing,Real}=nothing
    )
    
Optimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior!(
    la::AbstractLaplace;
    n_steps::Int=100,
    lr::Real=1e-1,
    λinit::Union{Nothing,Real}=nothing,
    σinit::Union{Nothing,Real}=nothing,
    verbose::Bool=false,
    tune_σ::Bool=la.likelihood == :regression,
)

    # Setup:
    logP₀ = isnothing(λinit) ? log.(unique(diag(la.P₀))) : log.([λinit])   # prior precision (scalar)
    logσ = isnothing(σinit) ? log.([la.σ]) : log.([σinit])                 # noise (scalar)
    opt = Adam(lr)
    show_every = round(n_steps / 10)
    i = 0
    if tune_σ
        @assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
        ps = Flux.params(logP₀, logσ)
    else
        if la.likelihood == :regression
            @warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
        end
        ps = Flux.params(logP₀)
    end
    loss(P₀, σ) = -log_marginal_likelihood(la; P₀=P₀[1], σ=σ[1])

    # Optimization:
    while i < n_steps
        gs = gradient(ps) do
            loss(exp.(logP₀), exp.(logσ))
        end
        update!(opt, ps, gs)
        i += 1
        if verbose
            if i % show_every == 0
                @info "Iteration $(i): P₀=$(exp(logP₀[1])), σ=$(exp(logσ[1]))"
                @show loss(exp.(logP₀), exp.(logσ))
                println("Log likelihood: $(log_likelihood(la))")
                println("Log det ratio: $(log_det_ratio(la))")
                println("Scatter: $(_weight_penalty(la))")
            end
        end
    end
end
