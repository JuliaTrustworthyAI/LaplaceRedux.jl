using .Curvature
using Flux
using Flux.Optimise: Adam
using LinearAlgebra
using Zygote

mutable struct Laplace
    model::Flux.Chain
    likelihood::Symbol
    subset_of_weights::Symbol
    hessian_structure::Symbol
    curvature::Union{Curvature.CurvatureInterface,Nothing}
    σ::Real
    μ₀::Real
    P₀::Union{AbstractMatrix,UniformScaling}
    H::Union{AbstractArray,Nothing}
    P::Union{AbstractArray,Nothing}
    Σ::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
    n_data::Union{Int,Nothing}
    loss::Real
end

using Parameters

@with_kw struct LaplaceParams 
    subset_of_weights::Symbol=:all
    hessian_structure::Symbol=:full
    backend::Symbol=:EmpiricalFisher
    σ::Real=1.0
    μ₀::Real=0.0
    λ::Real=1.0
    P₀::Union{Nothing,UniformScaling}=nothing
    loss::Real=0.0
end

"""
    Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)    

Wrapper function to prepare Laplace approximation.
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...) 

    # Load hyperparameters:
    args = LaplaceParams(;kwargs...)
    @assert !(args.σ != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."
    P₀ = isnothing(args.P₀) ? UniformScaling(args.λ) : args.P₀
    nn = model

    # Instantiate LA:
    la = Laplace(
        model, likelihood, 
        args.subset_of_weights, args.hessian_structure, nothing, 
        args.σ, args.μ₀, P₀, 
        nothing, nothing, nothing, nothing, nothing,
        args.loss
    )
    params = get_params(la)
    la.curvature = getfield(Curvature,args.backend)(nn,likelihood,params)   # curvature interface
    la.n_params = length(reduce(vcat, [vec(θ) for θ ∈ params]))             # number of params
    la.P₀ = la.P₀(la.n_params)

    # Sanity:
    if isa(la.P₀, AbstractMatrix)
        @assert all(size(la.P₀) .== la.n_params) "Dimensions of prior Hessian $(size(la.P₀)) do not align with number of parameters ($(la.n_params))"
    end

    return la

end

# Traits:
include("traits.jl")

"""
    hessian_approximation(la::Laplace, d)

Computes the local Hessian approximation at a single data `d`.
"""
function hessian_approximation(la::Laplace, d)
    loss, H = getfield(Curvature, la.hessian_structure)(la.curvature,d)
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
function fit!(la::Laplace, data; override::Bool=true)

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
    la.loss = loss      # Loss
    la.H = H            # Hessian
    la.P = H + la.P₀    # posterior precision
    la.Σ = inv(la.P)    # posterior covariance
    la.n_data = n_data  # number of observations
    
end

"""
    glm_predictive_distribution(la::Laplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::Laplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.curvature,X)
    fvar = predictive_variance(la,𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

"""
    predictive_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.

"""
function predictive_variance(la::Laplace,𝐉)
    N = size(𝐉, 1)
    fvar = map(n -> 𝐉[n,:]' * la.Σ * 𝐉[n,:], 1:N)
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
    (la::Laplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::Laplace)(X::AbstractArray; kwrgs...)
    return predict(la, X; kwrgs...)
end

"""
    optimize_prior_precision(la::Laplace; n_steps=100, lr=1e-1, init_prior_prec=1.)
    
Optimize the prior precision post-hoc through empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior(
    la::Laplace; 
    n_steps::Int=100, lr::Real=1e-1,
    λinit::Union{Nothing,Real}=nothing,
    σinit::Union{Nothing,Real}=nothing
)

    # Setup:
    P₀ = isnothing(λinit) ? la.P₀ : Diagonal(init_prior_prec)
    σ = isnothing(σinit) ? la.σ : σinit
    ps = Zygote.Params([P₀, σ])
    opt = Adam(lr)

    # Optimization:
    i = 0
    while i < n_steps
        gs = gradient(-log_marginal_likelihood(la; P₀=P₀, σ=σ), P₀, σ)
        update!(opt, ps, gs)
        i += 1
    end

    return P₀, σ
end
