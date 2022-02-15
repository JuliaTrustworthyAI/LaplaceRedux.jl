# Packages
using LinearAlgebra
include("optim.jl")

∑(vector)=sum(vector)

# Sigmoid function:
function 𝛔(a)
    trunc = 8.0 # truncation to avoid numerical over/underflow
    a = clamp.(a,-trunc,trunc)
    p = exp.(a)
    p = p ./ (1 .+ p)
    return p
end

# Negative log likelihood
function 𝓁(w,w_0,H_0,X,y)
    N = length(y)
    D = size(X)[2]
    μ = 𝛔(X*w)
    Δw = w-w_0
    return - ∑( y[n] * log(μ[n]) + (1-y[n]) * log(1-μ[n]) for n=1:N) + 1/2 * Δw'H_0*Δw
end

# Gradient:
function ∇𝓁(w,w_0,H_0,X,y)
    N = length(y)
    μ = 𝛔(X*w)
    Δw = w-w_0
    g = ∑((μ[n]-y[n]) * X[n,:] for n=1:N)
    return g + H_0*Δw
end

# Hessian:
function ∇∇𝓁(w,w_0,H_0,X,y)
    N = length(y)
    μ = 𝛔(X*w)
    H = ∑(μ[n] * (1-μ[n]) * X[n,:] * X[n,:]' for n=1:N)
    return H + H_0
end

# Main function:
struct BayesLogreg
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    𝐇₀::Any
    𝐇::AbstractArray
    Σ̂::Union{AbstractArray,Nothing}
end

function bayes_logreg(X,y;w_0=nothing,H_0=nothing,𝓁=𝓁,∇𝓁=∇𝓁,∇∇𝓁=∇∇𝓁,constant=true,λ=1,optim_options...)
    # Setup:
    if constant
        if !all(X[:,1] .== 1)
            X = hcat(ones(size(X)[1]), X)
        else
        end
    end
    N, D = size(X)
    if isnothing(w_0)
        w_0 = zeros(D)
    end
    if isnothing(H_0)
        H_0 = UniformScaling(λ)
    end
    
    # Model:
    w_map, H_map = newton(𝓁, w_0, ∇𝓁, ∇∇𝓁, (w_0=w_0, H_0=H_0, X=X, y=y), optim_options...) # fit the model (find mode of posterior distribution)
    Σ_map = inv(H_map) # inverse Hessian at the mode
    Σ_map = Symmetric(Σ_map) # to ensure matrix is Hermitian (i.e. avoid rounding issues)
    
    # Output:
    mod = BayesLogreg(w_map, Σ_map, H_0, H_map, Σ_map)
    return mod
end


#  ------------ Outer constructor methods: ------------
# Accessing fields:
μ(mod::BayesLogreg) = mod.μ
Σ(mod::BayesLogreg) = mod.Σ
# Coefficients:
coef(mod::BayesLogreg) = mod.μ 

# Sampling from posterior distribution:
using Distributions
sample_posterior(mod::BayesLogreg, n) = rand(MvNormal(mod.μ, mod.Σ),n)

# Predictive distribution:
function glm_predictive_distribution(mod::BayesLogreg, X::AbstractArray)
    μ = mod.μ # MAP mean vector
    Σ = mod.Σ # MAP covariance matrix
    if !isa(X, Matrix)
        # turn into matrix if necessary:
        X = reshape(X, 1, length(X))
    end
    if size(μ)[1] > size(X)[2]
        # add constant if necessary:
        X = hcat(ones(size(X)[1]),X)
    end
    # Predictions:
    ŷ = X*μ
    # Predictive variance
    σ̂ = [X[n,:]'Σ*X[n,:] for n=1:size(X)[1]]
    σ̂ = reshape(σ̂, size(ŷ)...)
    return ŷ, σ̂
end

# Posterior predictions:
function predict(mod::BayesLogreg, X)
    ŷ, σ̂ = glm_predictive_distribution(mod, X)
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

# Plugin estimate (MAP)
function plugin(mod::BayesLogreg, X)
    ŷ, σ̂ = glm_predictive_distribution(mod, X)
    p = Flux.σ.(ŷ)
    return p
end