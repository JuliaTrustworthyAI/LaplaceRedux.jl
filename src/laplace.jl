using Flux, LinearAlgebra, .Curvature

function laplace(model::Any; likelihood=:classification, subset_of_weights=:last_layer, hessian_structure=:full,backend=Curvature.EFInterface,λ=1e-5) 
    # Initialize:
    𝐇₀ = UniformScaling(λ)
    nn = model
    if likelihood==:classification
        loss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y)
    else
        loss(x, y) = Flux.Losses.mse(nn(x),y)
    end
    laplace_redux = LaplaceRedux(model, likelihood, subset_of_weights, hessian_structure, backend, 𝐇₀, nothing, nothing, nothing, loss)
    laplace_redux.n_params = length(reduce(vcat, [vec(θ) for θ ∈ get_params(laplace_redux)]))
    return laplace_redux
end

mutable struct LaplaceRedux
    model::Any
    likelihood::Symbol
    subset_of_weights::Symbol
    hessian_structure::Symbol
    backend::Any
    𝐇₀::Any
    𝐇::Union{AbstractArray,Nothing}
    Σ̂::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
    loss::Function
end

function get_params(𝑳::LaplaceRedux)
    nn = 𝑳.model
    n_layers = length(nn)
    𝚯 = [Flux.params(nn)[i] for i in 1:n_layers]
    if 𝑳.subset_of_weights == :last_layer
        𝚯 = [𝚯[2*n_layers-1],𝚯[2*n_layers]] # only get last parameters and constants
    end
    return 𝚯
end

function hessian_approximation(𝑳::LaplaceRedux, d)
    nn = 𝑳.model
    𝚯 = get_params(𝑳)
    curvature_interface = 𝑳.backend(nn,𝑳.loss,𝚯)
    𝐇 = getfield(Curvature, 𝑳.hessian_structure)(curvature_interface,d)
    return 𝐇
end

function fit!(𝑳::LaplaceRedux,data)
    
    if isnothing(𝑳.𝐇)
        𝐇 = zeros(𝑳.n_params,𝑳.n_params)
        for d in data
            𝐇 += hessian_approximation(𝑳, d)
        end
        𝑳.𝐇 = 𝐇 + 𝑳.𝐇₀ # posterior precision
        𝑳.Σ̂ = inv(𝑳.𝐇) # posterior covariance
    end
    
end

function glm_predictive_distribution(𝑳::LaplaceRedux, X::AbstractArray)
    𝐉, ŷ = getfield(Curvature, 𝑳.hessian_structure)(curvature_interface,X)
    σ̂ = predictive_variance(𝑳,𝐉)
    return ŷ, σ̂
end

function predictive_variance(𝑳::LaplaceRedux,𝐉)
end

# Posterior predictions:
function predict(𝑳::LaplaceRedux, X::AbstractArray, link_approx=:probit)
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

