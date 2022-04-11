using Flux, LinearAlgebra, .Curvature

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

"""
    Laplace(model::Any; loss_type=:logitbinarycrossentropy, subset_of_weights=:last_layer, hessian_structure=:full,backend=:EmpiricalFisher,λ=1)    

Wrapper function to prepare Laplace approximation.

# Examples

```julia-repl
using Flux, BayesLaplace
nn = Chain(Dense(2,1))
la = Laplace(nn)
```

"""
function Laplace(model::Any; loss_type=:logitbinarycrossentropy, subset_of_weights=:last_layer, hessian_structure=:full,backend=:EmpiricalFisher,λ=1) 
    # Initialize:
    H₀ = UniformScaling(λ)
    nn = model
    loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
    # Instantiate:
    la = Laplace(model, loss, subset_of_weights, hessian_structure, nothing, H₀, nothing, nothing, nothing)
    params = get_params(la)
    la.curvature = getfield(Curvature,backend)(nn,la.loss,params) # instantiate chosen curvature interface
    la.n_params = length(reduce(vcat, [vec(θ) for θ ∈ params]))
    return la
end

"""
    get_params(la::Laplace) 

Retrieves the desired (sub)set of model parameters and stores them in a list.

# Examples

```julia-repl
using Flux, BayesLaplace
nn = Chain(Dense(2,1))
la = Laplace(nn)
BayesLaplace.get_params(la)
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
using Flux, BayesLaplace
x, y = toy_data_linear()
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
    println(H)
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
    N = size(𝐉)[1]
    Σ = map(n -> 𝐉[n,:]' * la.Σ * 𝐉[n,:], 1:N)
    return Σ
end

# Posterior predictions:
"""
    predict(la::Laplace, X::AbstractArray; link_approx=:probit)

Computes predictions from Bayesian neural network.

# Examples

```julia-repl
using Flux, BayesLaplace
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
    # Probit approximation
    κ = 1 ./ sqrt.(1 .+ π/8 .* Σ) 
    z = κ .* ŷ
    # Truncation to avoid numerical over/underflow:
    trunc = 8.0 
    z = clamp.(z,-trunc,trunc)
    p = exp.(z)
    p = p ./ (1 .+ p)
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

