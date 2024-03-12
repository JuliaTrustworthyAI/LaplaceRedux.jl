using LinearAlgebra
using MLUtils

"Abstract base type for all Laplace approximations in this library"
abstract type BaseLaplace end
# NOTE: all subclasses implemented are parametric.
# If functional LA is implemented, it may make sense to add another layer of interface-inheritance

"""
Compile-time copy-paste macro @def: a macro that creates a macro with the specified name and content,
which is then immediately applied to the code.

Ref: https://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/
"""
macro def(name, definition)
    return quote
        macro $(esc(name))()
            return esc($(Expr(:quote, definition)))
        end
    end
end

@def fields_baselaplace begin
    model::Flux.Chain
    likelihood::Symbol
    subset_of_weights::Symbol
    # indices of the subnetwork
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}
    hessian_structure::Symbol
    curvature::Union{Curvature.CurvatureInterface,Nothing}
    # standard deviation in the Gaussian prior
    σ::Real
    # prior mean
    μ₀::Real
    # posterior mean
    μ::AbstractVector
    # prior precision (i.e. inverse covariance matrix)
    P₀::Union{AbstractMatrix,UniformScaling}
    # Hessian matrix
    H::Union{AbstractArray,KronDecomposed,Nothing}
    # posterior precision
    P::Union{AbstractArray,KronDecomposed,Nothing}
    # posterior covariance matrix
    Σ::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
    n_data::Union{Int,Nothing}
    n_out::Union{Int,Nothing}
    loss::Real
end

"""
    outdim(la::BaseLaplace)

Helper function to determine the output dimension, corresponding to the number of neurons 
on the last layer of the NN, of a `Flux.Chain` with Laplace approximation.
"""
outdim(la::BaseLaplace) = la.n_out

"""
    get_params(la::BaseLaplace) 

Retrieves the desired (sub)set of model parameters and stores them in a list.

# Examples

```julia-repl
using Flux, LaplaceRedux
# define a neural network with one hidden layer that takes a two-dimensional input and produces a one-dimensional output
nn = Chain(Dense(2,1))
la = Laplace(nn)
LaplaceRedux.get_params(la)
```

"""
function get_params(la::BaseLaplace)
    nn = la.model
    params = Flux.params(nn)
    n_elements = length(params)
    if la.subset_of_weights == :all || la.subset_of_weights == :subnetwork
        # get all parameters and constants in logitbinarycrossentropy
        params = [θ for θ in params]
    elseif la.subset_of_weights == :last_layer
        # Only get last layer parameters:
        # params[n_elements] is the bias vector of the last layer
        # params[n_elements-1] is the weight matrix of the last layer
        params = [params[n_elements - 1], params[n_elements]]
    end
    return params
end

@doc raw"""
    posterior_precision(la::BaseLaplace)

Computes the posterior precision ``P`` for a fitted Laplace Approximation as follows,

``
P = \sum_{n=1}^N\nabla_{\theta}^2\log p(\mathcal{D}_n|\theta)|_{\theta}_{MAP} + \nabla_{\theta}^2 \log p(\theta)|_{\theta}_{MAP} 
``

where ``\sum_{n=1}^N\nabla_{\theta}^2\log p(\mathcal{D}_n|\theta)|_{\theta}_{MAP}=H`` and ``\nabla_{\theta}^2 \log p(\theta)|_{\theta}_{MAP}=P_0``.
"""
function posterior_precision(la::BaseLaplace, H=la.H, P₀=la.P₀)
    @assert !isnothing(H) "Hessian not available. Either no value supplied or Laplace Approximation has not yet been estimated."
    return H + P₀
end

@doc raw"""
    posterior_covariance(la::BaseLaplace, P=la.P)

Computes the posterior covariance ``∑`` as the inverse of the posterior precision: ``\Sigma=P^{-1}``.
"""
function posterior_covariance(la::BaseLaplace, P=posterior_precision(la))
    @assert !isnothing(P) "Posterior precision not available. Either no value supplied or Laplace Approximation has not yet been estimated."
    return inv(P)
end

"""
    log_likelihood(la::BaseLaplace)


"""
function log_likelihood(la::BaseLaplace)
    factor = -_H_factor(la)
    if la.likelihood == :regression
        c = la.n_data * la.n_out * log(la.σ * sqrt(2 * pi))
    else
        c = 0
    end
    return factor * la.loss - c
end

"""
    _H_factor(la::BaseLaplace)

Returns the factor σ⁻², where σ is used in the zero-centered Gaussian prior p(θ) = N(θ;0,σ²I)
"""
_H_factor(la::BaseLaplace) = 1 / (la.σ^2)

"""
    _init_H(la::BaseLaplace)


"""
_init_H(la::BaseLaplace) = zeros(la.n_params, la.n_params)

"""
    _weight_penalty(la::BaseLaplace)

The weight penalty term is a regularization term used to prevent overfitting.
Weight regularization methods such as weight decay introduce a penalty to the loss function when training a neural network to encourage the network to use small weights.
Smaller weights in a neural network can result in a model that is more stable and less likely to overfit the training dataset, in turn having better performance when 
making a prediction on new data.
"""
function _weight_penalty(la::BaseLaplace)
    μ = la.μ                                                                 # MAP
    μ₀ = la.μ₀                                                               # prior
    Δ = μ .- μ₀
    return Δ'la.P₀ * Δ                                                       # measure of how far the MAP estimate deviates from the prior mean μ₀
end                                                                          # used to control the degree of regularization applied to the mode

"""
    log_marginal_likelihood(la::BaseLaplace; P₀::Union{Nothing,UniformScaling}=nothing, σ::Union{Nothing, Real}=nothing)


"""
function log_marginal_likelihood(
    la::BaseLaplace;
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
    log_det_ratio(la::BaseLaplace)


"""
function log_det_ratio(la::BaseLaplace)
    return log_det_posterior_precision(la) - log_det_prior_precision(la)
end

"""
    log_det_prior_precision(la::BaseLaplace)


"""
log_det_prior_precision(la::BaseLaplace) = sum(log.(diag(la.P₀)))

"""
    log_det_posterior_precision(la::BaseLaplace)


"""
log_det_posterior_precision(la::BaseLaplace) = logdet(posterior_precision(la))

"""
    hessian_approximation(la::BaseLaplace, d; batched::Bool=false)

Computes the local Hessian approximation at a single datapoint `d`.
"""
function hessian_approximation(la::BaseLaplace, d; batched::Bool=false)
    loss, H = getfield(Curvature, la.hessian_structure)(la.curvature, d; batched=batched)
    return loss, H
end

"""
    fit!(la::BaseLaplace,data)

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
function fit!(la::BaseLaplace, data; override::Bool=true)
    return _fit!(la, data; batched=false, batchsize=1, override=override)
end

"""
Fit the Laplace approximation, with batched data.
"""
function fit!(la::BaseLaplace, data::DataLoader; override::Bool=true)
    return _fit!(la, data; batched=true, batchsize=data.batchsize, override=override)
end

"""
    glm_predictive_distribution(la::BaseLaplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::BaseLaplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.curvature, X)
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

# Posterior predictions:
"""
    predict(la::BaseLaplace, X::AbstractArray; link_approx=:probit)

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
function predict(la::BaseLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true)
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
function predict(la::BaseLaplace, X::Matrix; link_approx=:probit, predict_proba::Bool=true)
    return stack([
        predict(la, X[:, i]; link_approx=link_approx, predict_proba=predict_proba) for i in 1:size(X, 2)
    ])
end

"""
    (la::BaseLaplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::BaseLaplace)(X::AbstractArray; kwrgs...)
    return predict(la, X; kwrgs...)
end

"""
    optimize_prior!(
        la::BaseLaplace; 
        n_steps::Int=100, lr::Real=1e-1,
        λinit::Union{Nothing,Real}=nothing,
        σinit::Union{Nothing,Real}=nothing
    )
    
Optimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior!(
    la::BaseLaplace;
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
