using LinearAlgebra

"""
    outdim(la::Laplace)

Helper function to determine the output dimension of a `Flux.Chain` with Laplace approximation.
"""
outdim(la::Laplace) = la.n_out

"""
    get_params(la::Laplace) 

Retrieves the desired (sub)set of model parameters and stores them in a list.

# Examples

```julia-repl
using Flux, LaplaceRedux
nn = Chain(Dense(2,1))
la = Laplace(nn)
LaplaceRedux.get_params(la)
```

"""
function get_params(la::Laplace)
    nn = la.model
    params = Flux.params(nn)
    n_elements = length(params)
    if la.subset_of_weights == :all
        params = [θ for θ ∈ params] # get all parameters and constants in logitbinarycrossentropy
    elseif la.subset_of_weights == :last_layer
        params = [params[n_elements-1],params[n_elements]] # only get last parameters and constants
    else
        @error "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer]`"
    end 
    return params
end

"""
    log_likelihood(la::Laplace)


"""
function log_likelihood(la::Laplace)
    factor = - _H_factor(la)
    if la.likelihood == :regression
        c = la.n_data * outdim(la) * log(la.σ * sqrt(2 * pi))
    else
        c = 0
    end
    return factor * la.loss - c
end

"""
    _H_factor(la::Laplace)


"""
_H_factor(la::Laplace) = 1 / (la.σ^2)

"""
    _init_H(la::Laplace)


"""
_init_H(la::Laplace) = zeros(la.n_params, la.n_params)

"""
    _weight_penalty(la::Laplace)


"""
function _weight_penalty(la::Laplace)
    μ = la.μ    # MAP
    μ₀ = la.μ₀  # prior
    Δ = μ .- μ₀
    return Δ'la.P₀*Δ
end

"""
    log_marginal_likelihood(la::Laplace; P₀::Union{Nothing,UniformScaling}=nothing, σ::Union{Nothing, Real}=nothing)


"""
function log_marginal_likelihood(la::Laplace; P₀::Union{Nothing,AbstractMatrix}=nothing, σ::Union{Nothing, Real}=nothing)

    # update prior precision:
    if !isnothing(P₀)
        la.P₀ = P₀
    end

    # update observation noise:
    if !isnothing(σ)
        @assert la.likelihood==:regression "Can only change sigma_noise for regression."
        la.σ = σ
    end

    mll = log_likelihood(la) - 0.5 * (log_det_ratio(la) + _weight_penalty(la))

    return mll
end

"""
    log_det_ratio(la::Laplace)


"""
log_det_ratio(la::Laplace) = log_det_posterior_precision(la) - log_det_prior_precision(la)

"""
    log_det_prior_precision(la::Laplace)


"""
log_det_prior_precision(la::Laplace) = logdet(la.P₀)

"""
    log_det_posterior_precision(la::Laplace)


"""
log_det_posterior_precision(la::Laplace) = logdet(la.P)


