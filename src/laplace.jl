using Flux, LinearAlgebra, .Curvature

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

function get_params(𝑳::LaplaceRedux)
    nn = 𝑳.model
    n_layers = length(nn)
    𝚯 = Flux.params(nn)
    if 𝑳.subset_of_weights == :last_layer
        𝚯 = [𝚯[2*n_layers-1],𝚯[2*n_layers]] # only get last parameters and constants
    end
    return 𝚯
end

function hessian_approximation(𝑳::LaplaceRedux, d)
    𝐇 = getfield(Curvature, 𝑳.hessian_structure)(𝑳.𝑪,d)
    return 𝐇
end

function fit!(𝑳::LaplaceRedux,data)

    𝐇 = zeros(𝑳.n_params,𝑳.n_params)
    for d in data
        𝐇 += hessian_approximation(𝑳, d)
    end
    𝑳.𝐇 = 𝐇 + 𝑳.𝐇₀ # posterior precision
    𝑳.Σ̂ = inv(𝑳.𝐇) # posterior covariance
    
end

function glm_predictive_distribution(𝑳::LaplaceRedux, X::AbstractArray)
    𝐉, ŷ = Curvature.jacobians(𝑳.𝑪,X)
    σ̂ = predictive_variance(𝑳,𝐉)
    σ̂ = reshape(σ̂, size(ŷ)...)
    return ŷ, σ̂
end

function predictive_variance(𝑳::LaplaceRedux,𝐉)
    N = size(𝐉)[1]
    σ̂ = map(n -> 𝐉[n,:]' * 𝑳.Σ̂ * 𝐉[n,:], 1:N)
    return σ̂
end

# Posterior predictions:
function predict(𝑳::LaplaceRedux, X::AbstractArray; link_approx=:probit)
    ŷ, σ̂ = glm_predictive_distribution(𝑳::LaplaceRedux, X::AbstractArray)
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

