using Flux

function laplace(model::Any; likelihood=:classification, subset_of_weights=:last_layer, hessian_structure=:full, backend=EFInterface) 
    laplace_redux = LaplaceRedux(model, likelihood, subset_of_weights, hessian_structure, backend)
    return laplace_redux
end
mutable struct LaplaceRedux
    model::Any
    likelihood::Symbol
    subset_of_weights::Symbol
    hessian_strcuture::Symbol
    backend::CurvatureInterface
    Σ̂::AbstractArray
end

function fit(𝑳::LaplaceRedux,data)
    for d in data
        𝐇 += hessian_approximation(𝑳, d)
    end
    return PosteriorPredictive(μ,Σ)
end

# Posterior predictions:
function predict(𝑳::LaplaceRedux, X::AbstractArray, link_approx=:probit)
    μ = mod.μ # MAP mean vector
    Σ = mod.Σ # MAP covariance matrix
    if !isa(X, Matrix)
        X = reshape(X, 1, length(X))
    end
    # Inner product:
    z = X*μ
    # Probit approximation
    v = [X[n,:]'Σ*X[n,:] for n=1:size(X)[1]]
    κ = 1 ./ sqrt.(1 .+ π/8 .* v) 
    z = κ .* z
    # Truncation to avoid numerical over/underflow:
    trunc = 8.0 
    z = clamp.(z,-trunc,trunc)
    p = exp.(z)
    p = p ./ (1 .+ p)
    return p
end

function hessian_approximation(𝑳::LaplaceRedux, d)
    nn = 𝑳.model
    n_layers = length(nn)
    𝚯 = [params(nn)[i] for i in 1:n_layers]
    if 𝑳.subset_of_weights == :last_layer
        𝚯 = [𝚯[n_layers]]
    end
    curvature_interface = 𝑳.backend(nn,𝚯)
    𝐇ᵢ = full(curvature_interface, d)
    return 𝐇ᵢ
end


