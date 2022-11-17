using .Curvature
using Flux
using LinearAlgebra

mutable struct Laplace
    model::Flux.Chain
    loss::Function
    subset_of_weights::Symbol
    hessian_structure::Symbol
    curvature::Union{Curvature.CurvatureInterface,Nothing}
    H₀::Any
    H::Union{AbstractArray,Nothing}
    Σ::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
end

using Parameters

@with_kw struct LaplaceParams 
    loss_type::Symbol=:logitbinarycrossentropy
    subset_of_weights::Symbol=:all
    hessian_structure::Symbol=:full
    backend::Symbol=:EmpiricalFisher
    λ::Real=1
    H₀::Union{Nothing, AbstractMatrix}=nothing
end

"""
    Laplace(model::Any; loss_type=:logitbinarycrossentropy, subset_of_weights=:last_layer, hessian_structure=:full,backend=:EmpiricalFisher,λ=1)    

Wrapper function to prepare Laplace approximation.

# Examples

```julia-repl
using Flux, LaplaceRedux
nn = Chain(Dense(2,1))
la = Laplace(nn)
```

"""
function Laplace(model::Any;kwargs...) 

    # Load hyperparameters:
    args = LaplaceParams(;kwargs...)

    # Prior:
    if isnothing(args.H₀)
        H₀ = UniformScaling(args.λ)
    else
        H₀ = args.H₀
    end

    # Model: 
    nn = model
    loss(x, y) = getfield(Flux.Losses,args.loss_type)(nn(x), y)

    # Instantiate:
    la = Laplace(model, loss, args.subset_of_weights, args.hessian_structure, nothing, H₀, nothing, nothing, nothing)
    params = get_params(la)
    la.curvature = getfield(Curvature,args.backend)(nn,la.loss,params) # instantiate chosen curvature interface
    la.n_params = length(reduce(vcat, [vec(θ) for θ ∈ params]))

    # Sanity:
    if isa(la.H₀, AbstractMatrix)
        @assert all(size(la.H₀) .== la.n_params) "Dimensions of prior Hessian $(size(la.H₀)) do not align with number of parameters ($(Fala.n_params))"
    end

    return la

end

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
    hessian_approximation(la::Laplace, d)

Computes the local Hessian approximation at a single data `d`.
"""
function hessian_approximation(la::Laplace, d)
    H = getfield(Curvature, la.hessian_structure)(la.curvature,d)
    return H
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
function fit!(la::Laplace,data)

    H = zeros(la.n_params,la.n_params)
    for d in data
        H += hessian_approximation(la, d)
    end
    la.H = H + la.H₀ # posterior precision
    la.Σ = inv(la.H) # posterior covariance
    
end

"""
    glm_predictive_distribution(la::Laplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::Laplace, X::AbstractArray)
    𝐉, ŷ = Curvature.jacobians(la.curvature,X)
    Σ = predictive_variance(la,𝐉)
    Σ = reshape(Σ, size(ŷ)...)
    return ŷ, Σ
end

"""
    predictive_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = H⁻¹`.

"""
function predictive_variance(la::Laplace,𝐉)
    N = size(𝐉, 1)
    Σ = map(n -> 𝐉[n,:]' * la.Σ * 𝐉[n,:], 1:N)
    return Σ
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
    ŷ, Σ = glm_predictive_distribution(la, X)

    # Regression:
    if la.loss == Flux.Losses.mse
        return ŷ, Σ
    end

    # Classification:
    if link_approx==:probit
        # Probit approximation
        κ = 1 ./ sqrt.(1 .+ π/8 .* Σ) 
        z = κ .* ŷ
        if size(z,1) == 1
            p = Flux.sigmoid(z)
        else
            p = Flux.softmax(z, dims=2)
        end
    end

    return p
end

"""
    plugin(la::Laplace, X::AbstractArray)

Computes the plugin estimate.
"""
function plugin(la::Laplace, X::AbstractArray)
    ŷ, Σ = glm_predictive_distribution(la, X)
    p = Flux.σ.(ŷ)
    return p
end

using Flux.Optimise: Adam
"""
    optimize_prior_precision(la::Laplace; n_steps=100, lr=1e-1, init_prior_prec=1.)
    
Optimize the prior precision post-hoc through empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior_precision(la::Laplace; n_steps=100, lr=1e-1, init_prior_prec=1.)
    la.H₀ = Diagonal(init_prior_prec)
    opt = Adam(lr)
    for i in 1:n_steps

    end
end
