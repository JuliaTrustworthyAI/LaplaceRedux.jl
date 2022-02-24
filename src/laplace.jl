using Flux, LinearAlgebra, .Curvature

"""
    laplace(model::Any; loss_type=:logitbinarycrossentropy, subset_of_weights=:last_layer, hessian_structure=:full,backend=:EmpiricalFisher,λ=1)    

Wrapper function to prepare Laplace approximation.

# Examples

```julia-repl
using Flux, BayesLaplace
nn = Chain(Dense(2,1))
la = laplace(nn)
```

"""
function laplace(model::Any; loss_type=:logitbinarycrossentropy, subset_of_weights=:last_layer, hessian_structure=:full,backend=:EmpiricalFisher,λ=1) 
    # Initialize:
    𝐇₀ = UniformScaling(λ)
    nn = model
    loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
    # Instantiate:
    𝑳 = LaplaceRedux(model, loss, subset_of_weights, hessian_structure, nothing, 𝐇₀, nothing, nothing, nothing)
    𝚯 = get_params(𝑳)
    𝑳.𝑪 = getfield(Curvature,backend)(nn,𝑳.loss,𝚯) # instantiate chosen curvature interface
    𝑳.n_params = length(reduce(vcat, [vec(θ) for θ ∈ 𝚯]))
    return 𝑳
end

mutable struct LaplaceRedux
    model::Any
    loss::Function
    subset_of_weights::Symbol
    hessian_structure::Symbol
    𝑪::Union{Curvature.CurvatureInterface,Nothing}
    𝐇₀::Any
    𝐇::Union{AbstractArray,Nothing}
    Σ̂::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
end

"""
    get_params(𝑳::LaplaceRedux) 

Retrieves the desired (sub)set of model parameters and stores them in a list.

# Examples

```julia-repl
using Flux, BayesLaplace
nn = Chain(Dense(2,1))
la = laplace(nn)
BayesLaplace.get_params(la)
```

"""
function get_params(𝑳::LaplaceRedux)
    nn = 𝑳.model
    𝚯 = Flux.params(nn)
    n_params = length(𝚯)
    if 𝑳.subset_of_weights == :all
        𝚯 = [θ for θ ∈ 𝚯] # get all parameters and constants in logitbinarycrossentropy
    elseif 𝑳.subset_of_weights == :last_layer
        𝚯 = [𝚯[n_params-1],𝚯[n_params]] # only get last parameters and constants
    else
        @error "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer]`"
    end 
    return 𝚯
end

"""
    hessian_approximation(𝑳::LaplaceRedux, d)

Computes the local Hessian approximation at a single data `d`.
"""
function hessian_approximation(𝑳::LaplaceRedux, d)
    𝐇 = getfield(Curvature, 𝑳.hessian_structure)(𝑳.𝑪,d)
    return 𝐇
end

"""
    fit!(𝑳::LaplaceRedux,data)

Fits the Laplace approximation for a data set.

# Examples

```julia-repl
using Flux, BayesLaplace
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = laplace(nn)
fit!(la, data)
```

"""
function fit!(𝑳::LaplaceRedux,data)

    𝐇 = zeros(𝑳.n_params,𝑳.n_params)
    for d in data
        𝐇 += hessian_approximation(𝑳, d)
    end
    𝑳.𝐇 = 𝐇 + 𝑳.𝐇₀ # posterior precision
    𝑳.Σ̂ = inv(𝑳.𝐇) # posterior covariance
    
end

"""
    glm_predictive_distribution(𝑳::LaplaceRedux, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(𝑳::LaplaceRedux, X::AbstractArray)
    𝐉, ŷ = Curvature.jacobians(𝑳.𝑪,X)
    σ̂ = predictive_variance(𝑳,𝐉)
    σ̂ = reshape(σ̂, size(ŷ)...)
    return ŷ, σ̂
end

"""
    predictive_variance(𝑳::LaplaceRedux,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ̂𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ̂ = 𝐇⁻¹`.

"""
function predictive_variance(𝑳::LaplaceRedux,𝐉)
    N = size(𝐉)[1]
    σ̂ = map(n -> 𝐉[n,:]' * 𝑳.Σ̂ * 𝐉[n,:], 1:N)
    return σ̂
end

# Posterior predictions:
"""
    predict(𝑳::LaplaceRedux, X::AbstractArray; link_approx=:probit)

Computes predictions from Bayesian neural network.

# Examples

```julia-repl
using Flux, BayesLaplace
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = laplace(nn)
fit!(la, data)
predict(la, hcat(x...))
```

"""
function predict(𝑳::LaplaceRedux, X::AbstractArray; link_approx=:probit)
    ŷ, σ̂ = glm_predictive_distribution(𝑳, X)
    # Probit approximation
    κ = 1 ./ sqrt.(1 .+ π/8 .* σ̂) 
    z = κ .* ŷ
    # Truncation to avoid numerical over/underflow:
    trunc = 8.0 
    z = clamp.(z,-trunc,trunc)
    p = exp.(z)
    p = p ./ (1 .+ p)
    return p
end

"""
    plugin(𝑳::LaplaceRedux, X::AbstractArray)

Computes the plugin estimate.
"""
function plugin(𝑳::LaplaceRedux, X::AbstractArray)
    ŷ, σ̂ = glm_predictive_distribution(𝑳, X)
    p = Flux.σ.(ŷ)
    return p
end

