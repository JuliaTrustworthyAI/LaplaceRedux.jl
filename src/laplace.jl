using .Curvature
using Flux
using Flux.Optimise: Adam, update!
using Flux.Optimisers: destructure
using LinearAlgebra
using MLUtils

mutable struct Laplace <: BaseLaplace
    model::Flux.Chain
    likelihood::Symbol
    subset_of_weights::Symbol
    hessian_structure::Symbol
    curvature::Union{Curvature.CurvatureInterface,Nothing}
    σ::Real
    μ₀::Real
    μ::AbstractVector
    P₀::Union{AbstractMatrix,UniformScaling}
    H::Union{AbstractArray,Nothing}
    P::Union{AbstractArray,Nothing}
    Σ::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
    n_data::Union{Int,Nothing}
    n_out::Union{Int,Nothing}
    loss::Real
end

using Parameters

@with_kw struct LaplaceParams
    subset_of_weights::Symbol = :all
    hessian_structure::Symbol = :full
    backend::Symbol = :EmpiricalFisher
    σ::Real = 1.0
    μ₀::Real = 0.0
    λ::Real = 1.0
    P₀::Union{Nothing,AbstractMatrix,UniformScaling} = nothing
    loss::Real = 0.0
end

"""
    Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)    

Wrapper function to prepare Laplace approximation.
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...)

    # Load hyperparameters:
    args = LaplaceParams(; kwargs...)

    # Assertions:
    @assert !(args.σ != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."
    @assert args.subset_of_weights ∈ [:all, :last_layer] "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer]`"

    # Setup:
    P₀ = isnothing(args.P₀) ? UniformScaling(args.λ) : args.P₀
    nn = model
    n_out = outdim(nn)
    μ = reduce(vcat, [vec(θ) for θ in Flux.params(nn)])

    # Instantiate LA:
    la = Laplace(
        model,
        likelihood,
        args.subset_of_weights,
        args.hessian_structure,
        nothing,
        args.σ,
        args.μ₀,
        μ,
        P₀,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        n_out,
        args.loss,
    )

    @assert outdim(la) == 1 "Support for multi-class output still lacking, sorry. Currently only regression and binary classification models are supported."

    params = get_params(la)
    la.curvature = getfield(Curvature, args.backend)(nn, likelihood, params)   # curvature interface
    la.n_params = length(reduce(vcat, [vec(θ) for θ in params]))             # number of params
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
    hessian_approximation(la::Laplace, d)

Computes the local Hessian approximation at a single datapoint `d`.
"""
function hessian_approximation(la::Laplace, d)
    loss, H = getfield(Curvature, la.hessian_structure)(la.curvature, d)
    return loss, H
end

"""
    fit!(la::Laplace,data)

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
function fit!(la::Laplace, data; override::Bool=true, batchsize=1)

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
    return la.n_data = n_data                  # number of observations
end

# TODO: code reuse
function fit!(la::Laplace, data::DataLoader; override::Bool=true)

    # FIXME MASSIVE HACK
    la.hessian_structure = :full_b

    if override
        H = _init_H(la)
        loss = 0.0
        n_data = 0
    end

    # Training:
    for d in data
        loss_batch, H_batch = hessian_approximation(la, d)
        loss += loss_batch
        # @show size(H_batch)
        # @show size(H)
        H += H_batch
        n_data += data.batchsize
    end

    # Store output:
    la.loss = loss                      # Loss
    la.H = H                            # Hessian
    la.P = posterior_precision(la)      # posterior precision
    la.Σ = posterior_covariance(la)     # posterior covariance
    return la.n_data = n_data                  # number of observations
end



"""
    glm_predictive_distribution(la::Laplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::Laplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.curvature, X)
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

"""
    functional_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.

"""
function functional_variance(la::Laplace, 𝐉)
    Σ = posterior_covariance(la)
    fvar = map(j -> (j' * Σ * j), eachcol(𝐉))
    return fvar
end

# Posterior predictions:
"""
    predict(la::Laplace, X::AbstractArray; link_approx=:probit)

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
function predict(la::Laplace, X::AbstractArray; link_approx=:probit)
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
        if outdim(la) == 1
            p = Flux.sigmoid(z)
        else
            p = Flux.softmax(z; dims=1)
        end

        return p
    end
end

"""
    (la::Laplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::Laplace)(X::AbstractArray; kwrgs...)
    return predict(la, X; kwrgs...)
end

"""
    optimize_prior!(
        la::Laplace; 
        n_steps::Int=100, lr::Real=1e-1,
        λinit::Union{Nothing,Real}=nothing,
        σinit::Union{Nothing,Real}=nothing
    )
    
Optimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior!(
    la::Laplace;
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

    # la.P = la.H + la.P₀
    # la.Σ = inv(la.P)

end
