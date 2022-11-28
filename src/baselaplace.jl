using LinearAlgebra

"Abstract base type of Laplace Approximation."
abstract type BaseLaplace end

"""
    outdim(la::BaseLaplace)

Helper function to determine the output dimension of a `Flux.Chain` with Laplace approximation.
"""
outdim(la::BaseLaplace) = la.n_out

"""
    get_params(la::BaseLaplace) 

Retrieves the desired (sub)set of model parameters and stores them in a list.

# Examples

```julia-repl
using Flux, LaplaceRedux
nn = Chain(Dense(2,1))
la = Laplace(nn)
LaplaceRedux.get_params(la)
```

"""
function get_params(la::BaseLaplace)
    nn = la.model
    params = Flux.params(nn)
    n_elements = length(params)
    if la.subset_of_weights == :all
        params = [θ for θ ∈ params]                         # get all parameters and constants in logitbinarycrossentropy
    elseif la.subset_of_weights == :last_layer
        params = [params[n_elements-1],params[n_elements]]  # only get last parameters and constants
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

Computes the posterior covariance as the inverse of the posterior precision: ``\Sigma=P^{-1}``.
"""
function posterior_covariance(la::BaseLaplace, P=posterior_precision(la))
    @assert !isnothing(P) "Posterior precision not available. Either no value supplied or Laplace Approximation has not yet been estimated."
    return inv(P)
end


"""
    log_likelihood(la::BaseLaplace)


"""
function log_likelihood(la::BaseLaplace)
    factor = - _H_factor(la)
    if la.likelihood == :regression
        c = la.n_data * la.n_out * log(la.σ * sqrt(2 * pi))
    else
        c = 0
    end
    return factor * la.loss - c
end

"""
    _H_factor(la::BaseLaplace)


"""
_H_factor(la::BaseLaplace) = 1 / (la.σ^2)

"""
    _init_H(la::BaseLaplace)


"""
_init_H(la::BaseLaplace) = zeros(la.n_params, la.n_params)

"""
    _weight_penalty(la::BaseLaplace)


"""
function _weight_penalty(la::BaseLaplace)
    μ = la.μ    # MAP
    μ₀ = la.μ₀  # prior
    Δ = μ .- μ₀
    return Δ'la.P₀*Δ
end

"""
    log_marginal_likelihood(la::BaseLaplace; P₀::Union{Nothing,UniformScaling}=nothing, σ::Union{Nothing, Real}=nothing)


"""
function log_marginal_likelihood(la::BaseLaplace; P₀::Union{Nothing,AbstractFloat,AbstractMatrix}=nothing, σ::Union{Nothing, Real}=nothing)

    # update prior precision:
    if !isnothing(P₀)
        la.P₀ = typeof(P₀) <: AbstractFloat ? UniformScaling(P₀)(la.n_params) : P₀
    end

    # update observation noise:
    if !isnothing(σ)
        @assert (la.likelihood==:regression || la.σ == σ) "Can only change observational noise σ for regression."
        la.σ = σ
    end 

    return log_likelihood(la) - 0.5 * (log_det_ratio(la) + _weight_penalty(la))
end

"""
    log_det_ratio(la::BaseLaplace)


"""
log_det_ratio(la::BaseLaplace) = log_det_posterior_precision(la) - log_det_prior_precision(la)

"""
    log_det_prior_precision(la::Laplace)


"""
log_det_prior_precision(la::BaseLaplace) = sum(log.(diag(la.P₀)))

"""
    log_det_posterior_precision(la::Laplace)


"""
log_det_posterior_precision(la::BaseLaplace) = logdet(posterior_precision(la))


