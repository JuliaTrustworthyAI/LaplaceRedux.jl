"""
    Flux.params(la::Laplace)

Overloads the `params` function for a `Laplace` object.
"""
Flux.params(la::Laplace) = Flux.params(la.model, la.est_params)

"""
    LaplaceRedux.n_params(la::Laplace)

Overloads the `n_params` function for a `Laplace` object.
"""
LaplaceRedux.n_params(la::Laplace) = LaplaceRedux.n_params(la.model, la.est_params)

"""
    get_prior_mean(la::Laplace)

Helper function to extract the prior mean of the parameters from a Laplace approximation.
"""
function get_prior_mean(la::Laplace)
    return la.prior.μ₀
end

"""
    prior_precision(la::Laplace)

Helper function to extract the prior precision matrix from a Laplace approximation.
"""
function prior_precision(la::Laplace)
    return la.prior.P₀
end

"""
    outdim(la::AbstractLaplace)

Helper function to determine the output dimension, corresponding to the number of neurons 
on the last layer of the NN, of a `Flux.Chain` with Laplace approximation.
"""
outdim(la::AbstractLaplace) = outdim(la.model)

@doc raw"""
    posterior_precision(la::AbstractLaplace, H=la.posterior.H, P₀=la.prior.P₀)

Computes the posterior precision ``P`` for a fitted Laplace Approximation as follows,

``P = \sum_{n=1}^N\nabla_{\theta}^2 \log p(\mathcal{D}_n|\theta)|_{\hat\theta} + \nabla_{\theta}^2 \log p(\theta)|_{\hat\theta}``

where ``\sum_{n=1}^N\nabla_{\theta}^2\log p(\mathcal{D}_n|\theta)|_{\hat\theta}=H`` is the Hessian and ``\nabla_{\theta}^2 \log p(\theta)|_{\hat\theta}=P_0`` is the prior precision and ``\hat\theta`` is the MAP estimate.
"""
function posterior_precision(la::AbstractLaplace, H=la.posterior.H, P₀=la.prior.P₀)
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
        c = la.posterior.n_data * la.posterior.n_out * log(la.prior.σ * sqrt(2 * pi))
    else
        c = 0
    end
    return factor * la.posterior.loss - c
end

"""
    _H_factor(la::AbstractLaplace)

Returns the factor σ⁻², where σ is used in the zero-centered Gaussian prior p(θ) = N(θ;0,σ²I)
"""
_H_factor(la::AbstractLaplace) = 1 / (la.prior.σ^2)

"""
    _init_H(la::AbstractLaplace)


"""
_init_H(la::AbstractLaplace) = zeros(la.posterior.n_params, la.posterior.n_params)

"""
    _weight_penalty(la::AbstractLaplace)

The weight penalty term is a regularization term used to prevent overfitting.
Weight regularization methods such as weight decay introduce a penalty to the loss function when training a neural network to encourage the network to use small weights.
Smaller weights in a neural network can result in a model that is more stable and less likely to overfit the training dataset, in turn having better performance when 
making a prediction on new data.
"""
function _weight_penalty(la::AbstractLaplace)
    μ = la.posterior.μ
    μ₀ = get_prior_mean(la)
    Δ = μ .- μ₀
    P₀ = prior_precision(la)
    return Δ'P₀ * Δ
end

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
        la.prior.P₀ =
            typeof(P₀) <: AbstractFloat ? UniformScaling(P₀)(la.posterior.n_params) : P₀
    end

    # update observation noise:
    if !isnothing(σ)
        @assert (la.likelihood == :regression || la.prior.σ == σ) "Can only change observational noise σ for regression."
        la.prior.σ = σ
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
log_det_prior_precision(la::AbstractLaplace) = sum(log.(diag(la.prior.P₀)))

"""
    log_det_posterior_precision(la::AbstractLaplace)


"""
log_det_posterior_precision(la::AbstractLaplace) = logdet(posterior_precision(la))
