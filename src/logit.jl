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
end
function bayes_logreg(X,y;w_0=nothing,H_0=nothing,𝓁=𝓁,∇𝓁=∇𝓁,∇∇𝓁=∇∇𝓁,constant=true,λ=0.005,optim_options...)
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
    mod = BayesLogreg(w_map, Σ_map)
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
# Posterior predictions:
function predict(mod::BayesLogreg, X)
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

using AlgorithmicRecourse

"""
    retrain(𝑴::AlgorithmicRecourse.Models.BayesianLogisticModel, data; n_epochs=10) 

Retrains a fitted deep ensemble for (new) data.
"""
function retrain(𝑴::AlgorithmicRecourse.Models.LogisticModel, data; n_epochs=10, τ=1.0) 
    X = Flux.stack(map(d -> d[1], data),1)
    y = Flux.stack(map(d -> d[2], data),1)
    model = bayes_logreg(X, y)
    𝑴 = AlgorithmicRecourse.Models.LogisticModel(reshape(model.μ[2:end],1,length(model.μ)-1), [model.μ[1]]);
    return 𝑴
end

"""
    retrain(𝑴::AlgorithmicRecourse.Models.BayesianLogisticModel, data; n_epochs=10) 

Retrains a fitted deep ensemble for (new) data.
"""
function retrain(𝑴::AlgorithmicRecourse.Models.BayesianLogisticModel, data; n_epochs=10, τ=1.0) 
    X = Flux.stack(map(d -> d[1], data),1)
    y = Flux.stack(map(d -> d[2], data),1)
    model = bayes_logreg(X, y)
    𝑴 = AlgorithmicRecourse.Models.BayesianLogisticModel(reshape(model.μ,1,length(model.μ)), model.Σ);
    return 𝑴
end