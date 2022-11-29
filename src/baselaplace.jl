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
    log_det_prior_precision(la::BaseLaplace)


"""
log_det_prior_precision(la::BaseLaplace) = sum(log.(diag(la.P₀)))

"""
    log_det_posterior_precision(la::BaseLaplace)


"""
log_det_posterior_precision(la::BaseLaplace) = logdet(posterior_precision(la))

"""
    hessian_approximation(la::BaseLaplace, d)

Computes the local Hessian approximation at a single data `d`.
"""
function hessian_approximation(la::BaseLaplace, d)
    loss, H = getfield(Curvature, la.hessian_structure)(la.curvature,d)
    return loss, H
end

"""
    fit!(la::BaseLaplace,data)

Fits the Laplace approximation for a data set.

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

    if override
        H = _init_H(la)
        loss = 0.0
        n_data = 0
    end

    # Training:
    for d in data
        loss_batch, H_batch = hessian_approximation(la, d)
        loss += loss_batch
        H += H_batch
        n_data += 1
    end

    # Store output:
    la.loss = loss                      # Loss
    la.H = H                            # Hessian
    la.P = posterior_precision(la)      # posterior precision
    la.Σ = posterior_covariance(la)     # posterior covariance
    la.n_data = n_data                  # number of observations
    
end

"""
    glm_predictive_distribution(la::BaseLaplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::BaseLaplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.curvature,X)
    fvar = functional_variance(la,𝐉)
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
function predict(la::BaseLaplace, X::AbstractArray; link_approx=:probit)
    fμ, fvar = glm_predictive_distribution(la, X)

    # Regression:
    if la.likelihood == :regression
        return fμ, fvar
    end

    # Classification:
    if la.likelihood == :classification
        
        # Probit approximation
        if link_approx==:probit
            κ = 1 ./ sqrt.(1 .+ π/8 .* fvar) 
            z = κ .* fμ
        end

        if link_approx==:plugin
            z = fμ
        end

        # Sigmoid/Softmax
        if outdim(la) == 1
            p = Flux.sigmoid(z)
        else
            p = Flux.softmax(z, dims=1)
        end

        return p
    end
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
    n_steps::Int=100, lr::Real=1e-1,
    λinit::Union{Nothing,Real}=nothing,
    σinit::Union{Nothing,Real}=nothing,
    verbose::Bool=false,
    tune_σ::Bool=la.likelihood==:regression
)

    # Setup:
    logP₀ = isnothing(λinit) ? log.(unique(diag(la.P₀))) : log.([λinit])   # prior precision (scalar)
    logσ = isnothing(σinit) ? log.([la.σ]) : log.([σinit])                 # noise (scalar)
    opt = Adam(lr)
    show_every = round(n_steps/10)
    i = 0
    if tune_σ
        @assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
        ps = Flux.params(logP₀,logσ)
    else
        if la.likelihood == :regression
            @warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
        end
        ps = Flux.params(logP₀)
    end
    loss(P₀,σ) = - log_marginal_likelihood(la; P₀=P₀[1], σ=σ[1])

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


