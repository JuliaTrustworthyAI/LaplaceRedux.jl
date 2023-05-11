using .Curvature
using Flux
using Flux.Optimise: Adam, update!
using Flux.Optimisers: destructure
using LinearAlgebra

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
    subset_of_weights::Symbol=:all
    hessian_structure::Symbol=:full
    backend::Symbol=:EmpiricalFisher
    σ::Real=1.0
    μ₀::Real=0.0
    λ::Real=1.0
    P₀::Union{Nothing,AbstractMatrix,UniformScaling}=nothing
    loss::Real=0.0
end

"""
    Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)    

Wrapper function to prepare Laplace approximation.
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...) 

    # Load hyperparameters:
    args = LaplaceParams(;kwargs...)

    # Assertions:
    @assert !(args.σ != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."
    @assert args.subset_of_weights ∈ [:all, :last_layer] "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer]`"

    # Setup:
    P₀ = isnothing(args.P₀) ? UniformScaling(args.λ) : args.P₀
    nn = model
    n_out = outdim(nn)
    μ = reduce(vcat, [vec(θ) for θ ∈ Flux.params(nn)])

    # Instantiate LA:
    la = Laplace(
        model, likelihood, 
        args.subset_of_weights, args.hessian_structure, nothing, 
        args.σ, args.μ₀, μ, P₀, 
        nothing, nothing, nothing, nothing, nothing,
        n_out, args.loss
    )

    # @assert outdim(la)==1 "Support for multi-class output still lacking, sorry. Currently only regression and binary classification models are supported."

    params = get_params(la)
    la.curvature = getfield(Curvature,args.backend)(nn,likelihood,params)   # curvature interface
    la.n_params = length(reduce(vcat, [vec(θ) for θ ∈ params]))             # number of params
    la.μ = la.μ[(end-la.n_params+1):end]                                    # adjust weight vector
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
    la.loss = loss                      # Loss
    la.H = H                            # Hessian
    la.P = posterior_precision(la)      # posterior precision
    la.Σ = posterior_covariance(la)     # posterior covariance
    la.n_data = n_data                  # number of observations
    
end

"""
    glm_predictive_distribution(la::Laplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::Laplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.curvature,X)
    fvar = functional_variance(la,𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

"""
    functional_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.

"""
function functional_variance(la::Laplace,𝐉)
    Σ = posterior_covariance(la)
    #Σ = get_posterior_covariance_py()
    fvar = map(j -> (j' * Σ * j), eachcol(𝐉))
    return fvar
end

function get_posterior_covariance_py()
    py_posterior_cov = [[ 3.3274e+01, -2.7266e+00,  8.4388e-03,  1.8540e-02, -1.1284e-03,
         -2.1287e-04, -3.8825e-01,  5.2650e-03, -2.2294e-04,  1.2339e-02,
          9.1308e-02, -1.0260e-01,  1.0660e-01, -5.3066e-02,  3.3628e-02,
         -3.9588e-01,  3.9831e-01, -3.1048e-01,  2.7693e-01, -4.3659e-01,
          3.7945e-01,  9.4416e-02,  5.4464e-02, -5.9225e-02, -8.9692e-02],
        [-2.7266e+00,  2.3220e+01,  1.9330e-02,  1.1781e-01, -1.9358e-03,
         -1.1458e-03, -1.8955e+00,  1.4200e-02, -6.6012e-04, -1.6216e-02,
          4.7016e-01, -2.0177e-01, -1.0111e-01,  1.4675e-01,  2.1282e-01,
         -4.6803e-01,  9.4857e-01, -6.6480e-01,  5.8532e-01, -1.5656e+00,
          6.5376e-01,  3.9921e-01,  4.9346e-02,  2.5106e-01, -6.9977e-01],
        [ 8.4388e-03,  1.9330e-02,  3.4056e+01,  3.9143e-02, -1.7379e-04,
          1.5375e-05,  4.4017e-03, -8.3267e-03, -4.7615e-05,  8.3731e-03,
         -4.4691e-02, -5.0051e-02, -3.9282e-02,  7.8157e-02,  3.3268e-02,
          3.0069e-02, -3.7937e-02, -3.1201e-03,  8.4038e-04,  4.4699e-03,
          1.9902e-02, -3.1802e-02,  3.9319e-02, -1.1997e-02,  4.4789e-03],
        [ 1.8540e-02,  1.1781e-01,  3.9143e-02,  3.3889e+01, -1.6001e-05,
         -2.4928e-04,  1.8109e-02,  3.3265e-02,  1.3256e-05, -4.2082e-02,
          1.0711e-01,  2.5292e-01,  1.4193e-01, -3.0209e-01, -1.4153e-01,
         -6.5462e-02,  1.5390e-01, -2.8901e-02, -3.4389e-02,  4.1080e-02,
         -8.2486e-02,  8.9268e-02, -1.6284e-01,  7.1301e-02,  2.2765e-03],
        [-1.1284e-03, -1.9358e-03, -1.7379e-04, -1.6001e-05,  3.4052e+01,
         -7.6579e-04,  2.0983e-04, -1.4088e-04, -1.8089e-04, -9.2011e-03,
          1.8615e-02, -1.6306e-02,  2.4484e-02, -3.4714e-03, -2.0960e-02,
          6.3426e-03, -1.8383e-02,  2.3936e-02, -2.1625e-02,  3.2398e-03,
          1.3330e-02,  1.1378e-02,  2.0842e-02, -1.3848e-02, -1.8370e-02],
        [-2.1287e-04, -1.1458e-03,  1.5375e-05, -2.4928e-04, -7.6579e-04,
          3.4065e+01, -1.8571e-04,  2.6109e-07, -9.7910e-04, -4.2516e-04,
          7.3506e-03, -1.3440e-02, -3.0508e-03, -2.6473e-03,  1.5677e-02,
         -4.5657e-04, -5.3951e-03,  1.4884e-02,  3.9321e-03,  6.9198e-04,
         -1.7122e-02,  7.7192e-03, -5.6849e-03, -6.6078e-03,  4.5734e-03],
        [-3.8825e-01, -1.8955e+00,  4.4017e-03,  1.8109e-02,  2.0983e-04,
         -1.8571e-04,  3.3602e+01,  4.6404e-03, -2.0688e-04, -3.9687e-02,
          1.5119e-01,  4.7603e-02, -1.6761e-01,  1.0735e-01,  6.6709e-02,
          1.9709e-01,  9.1208e-02, -3.2212e-03,  1.0209e-02, -3.4977e-01,
         -1.1108e-01,  1.0223e-01, -5.9916e-02,  2.4552e-01, -2.8785e-01],
        [ 5.2650e-03,  1.4200e-02, -8.3267e-03,  3.3265e-02, -1.4088e-04,
          2.6109e-07,  4.6404e-03,  3.4059e+01, -3.9973e-05, -1.2262e-03,
         -3.1253e-02, -3.6876e-02, -2.2509e-02,  5.5042e-02,  2.5024e-02,
          1.8878e-02, -2.7282e-02, -3.4866e-03,  4.8561e-03,  3.4912e-03,
          1.5337e-02, -2.8734e-02,  3.3058e-02, -1.1936e-02,  7.6090e-03],
        [-2.2294e-04, -6.6012e-04, -4.7615e-05,  1.3256e-05, -1.8089e-04,
         -9.7910e-04, -2.0688e-04, -3.9973e-05,  3.4066e+01,  1.0219e-04,
          5.0625e-03, -7.0023e-03, -3.5060e-03, -1.2676e-03,  3.9209e-03,
         -1.3018e-03, -4.1619e-03,  6.3433e-03,  4.7055e-03,  3.6711e-04,
         -3.2619e-03,  5.7914e-03, -4.7529e-03, -6.0737e-03,  5.0351e-03],
        [ 1.2339e-02, -1.6216e-02,  8.3731e-03, -4.2082e-02, -9.2011e-03,
         -4.2516e-04, -3.9687e-02, -1.2262e-03,  1.0219e-04,  3.0220e+01,
          1.2269e+00,  6.0486e-01,  1.4705e+00, -3.9037e-01, -4.1223e-01,
          7.6472e-01, -7.1577e-01,  4.3990e-01,  1.6114e+00, -1.2078e-01,
         -6.3255e-01, -2.5189e+00,  1.0795e+00, -4.5586e-02,  1.4849e+00],
        [ 9.1308e-02,  4.7016e-01, -4.4691e-02,  1.0711e-01,  1.8615e-02,
          7.3506e-03,  1.5119e-01, -3.1253e-02,  5.0625e-03,  1.2269e+00,
          2.3801e+01, -2.2721e+00, -8.2082e-01,  2.8325e+00,  2.0655e+00,
         -5.5333e-01,  6.5400e+00, -1.1846e+00,  1.4720e-01,  8.9345e-01,
          1.3915e+00, -8.9200e+00,  2.0116e+00,  5.8889e+00,  1.0199e+00],
        [-1.0260e-01, -2.0177e-01, -5.0051e-02,  2.5292e-01, -1.6306e-02,
         -1.3440e-02,  4.7603e-02, -3.6876e-02, -7.0023e-03,  6.0486e-01,
         -2.2721e+00,  2.0960e+01,  3.7895e-02,  1.9443e+00,  2.1678e+00,
         -1.7726e+00,  6.3930e-01,  9.4700e+00,  1.1298e+00, -3.1135e-01,
          1.4690e+00, -1.8044e+00,  1.9848e+00, -1.0108e+00,  8.3061e-01],
        [ 1.0660e-01, -1.0111e-01, -3.9282e-02,  1.4193e-01,  2.4484e-02,
         -3.0508e-03, -1.6761e-01, -2.2509e-02, -3.5060e-03,  1.4705e+00,
         -8.2082e-01,  3.7895e-02,  2.2308e+01,  3.6671e+00, -1.3187e+00,
          4.2446e+00, -2.4512e+00,  5.3293e-01,  6.0429e+00, -3.9507e-01,
          7.4778e-01,  6.4080e-01, -8.1015e+00,  1.7980e+00,  5.6624e+00],
        [-5.3066e-02,  1.4675e-01,  7.8157e-02, -3.0209e-01, -3.4714e-03,
         -2.6473e-03,  1.0735e-01,  5.5042e-02, -1.2676e-03, -3.9037e-01,
          2.8325e+00,  1.9443e+00,  3.6671e+00,  2.6580e+01, -2.8913e+00,
         -2.8726e+00,  4.3056e+00,  2.6392e+00, -4.0408e-01,  3.4914e-01,
         -1.6920e+00,  2.4483e+00, -3.8118e+00,  1.4333e+00, -6.9457e-02],
        [ 3.3628e-02,  2.1282e-01,  3.3268e-02, -1.4153e-01, -2.0960e-02,
          1.5677e-02,  6.6709e-02,  2.5024e-02,  3.9209e-03, -4.1223e-01,
          2.0655e+00,  2.1678e+00, -1.3187e+00, -2.8913e+00,  1.8777e+01,
          2.6263e+00,  5.5376e-01,  3.4154e+00, -8.9540e-01,  2.7226e-01,
          9.7069e+00,  1.6512e+00, -4.2460e+00,  3.1990e+00, -6.0412e-01],
        [-3.9588e-01, -4.6803e-01,  3.0069e-02, -6.5462e-02,  6.3426e-03,
         -4.5657e-04,  1.9709e-01,  1.8878e-02, -1.3018e-03,  7.6472e-01,
         -5.5333e-01, -1.7726e+00,  4.2446e+00, -2.8726e+00,  2.6263e+00,
          2.6392e+01,  3.5898e+00, -2.8745e+00,  2.6652e+00, -1.6394e-01,
          2.0207e+00,  1.0287e-01,  1.3854e+00, -3.9955e+00,  2.5071e+00],
        [ 3.9831e-01,  9.4857e-01, -3.7937e-02,  1.5390e-01, -1.8383e-02,
         -5.3951e-03,  9.1208e-02, -2.7282e-02, -4.1619e-03, -7.1577e-01,
          6.5400e+00,  6.3930e-01, -2.4512e+00,  4.3056e+00,  5.5376e-01,
          3.5898e+00,  2.2495e+01, -1.2728e+00, -4.2282e-01,  7.2682e-01,
          8.0130e-02,  5.7544e+00,  1.8467e+00, -7.8918e+00,  2.9123e-01],
        [-3.1048e-01, -6.6480e-01, -3.1201e-03, -2.8901e-02,  2.3936e-02,
          1.4884e-02, -3.2212e-03, -3.4866e-03,  6.3433e-03,  4.3990e-01,
         -1.1846e+00,  9.4700e+00,  5.3293e-01,  2.6392e+00,  3.4154e+00,
         -2.8745e+00, -1.2728e+00,  1.8887e+01,  1.9016e+00, -1.8142e-01,
          2.2948e+00, -6.0287e-01,  3.1768e+00, -4.2990e+00,  1.7252e+00],
        [ 2.7693e-01,  5.8532e-01,  8.4038e-04, -3.4389e-02, -2.1625e-02,
          3.9321e-03,  1.0209e-02,  4.8561e-03,  4.7055e-03,  1.6114e+00,
          1.4720e-01,  1.1298e+00,  6.0429e+00, -4.0408e-01, -8.9540e-01,
          2.6652e+00, -4.2282e-01,  1.9016e+00,  2.3747e+01,  6.7981e-01,
         -2.1359e+00,  1.7751e+00,  5.6364e+00,  2.2430e+00, -9.6546e+00],
        [-4.3659e-01, -1.5656e+00,  4.4699e-03,  4.1080e-02,  3.2398e-03,
          6.9198e-04, -3.4977e-01,  3.4912e-03,  3.6711e-04, -1.2078e-01,
          8.9345e-01, -3.1135e-01, -3.9507e-01,  3.4914e-01,  2.7226e-01,
         -1.6394e-01,  7.2682e-01, -1.8142e-01,  6.7981e-01,  3.2097e+01,
          2.2055e-01,  7.1779e-01, -4.6182e-02,  5.6998e-01, -1.2415e+00],
        [ 3.7945e-01,  6.5376e-01,  1.9902e-02, -8.2486e-02,  1.3330e-02,
         -1.7122e-02, -1.1108e-01,  1.5337e-02, -3.2619e-03, -6.3255e-01,
          1.3915e+00,  1.4690e+00,  7.4778e-01, -1.6920e+00,  9.7069e+00,
          2.0207e+00,  8.0130e-02,  2.2948e+00, -2.1359e+00,  2.2055e-01,
          2.0596e+01,  7.5630e-01, -9.1553e-01,  2.1110e+00, -1.9517e+00],
        [ 9.4416e-02,  3.9921e-01, -3.1802e-02,  8.9268e-02,  1.1378e-02,
          7.7192e-03,  1.0223e-01, -2.8734e-02,  5.7914e-03, -2.5189e+00,
         -8.9200e+00, -1.8044e+00,  6.4080e-01,  2.4483e+00,  1.6512e+00,
          1.0287e-01,  5.7544e+00, -6.0287e-01,  1.7751e+00,  7.1779e-01,
          7.5630e-01,  2.2717e+01,  3.0885e+00,  5.7846e+00,  2.4769e+00],
        [ 5.4464e-02,  4.9346e-02,  3.9319e-02, -1.6284e-01,  2.0842e-02,
         -5.6849e-03, -5.9916e-02,  3.3058e-02, -4.7529e-03,  1.0795e+00,
          2.0116e+00,  1.9848e+00, -8.1015e+00, -3.8118e+00, -4.2460e+00,
          1.3854e+00,  1.8467e+00,  3.1768e+00,  5.6364e+00, -4.6182e-02,
         -9.1553e-01,  3.0885e+00,  2.2149e+01,  3.2378e+00,  5.5917e+00],
        [-5.9225e-02,  2.5106e-01, -1.1997e-02,  7.1301e-02, -1.3848e-02,
         -6.6078e-03,  2.4552e-01, -1.1936e-02, -6.0737e-03, -4.5586e-02,
          5.8889e+00, -1.0108e+00,  1.7980e+00,  1.4333e+00,  3.1990e+00,
         -3.9955e+00, -7.8918e+00, -4.2990e+00,  2.2430e+00,  5.6998e-01,
          2.1110e+00,  5.7846e+00,  3.2378e+00,  2.2241e+01,  2.8032e+00],
        [-8.9692e-02, -6.9977e-01,  4.4789e-03,  2.2765e-03, -1.8370e-02,
          4.5734e-03, -2.8785e-01,  7.6090e-03,  5.0351e-03,  1.4849e+00,
          1.0199e+00,  8.3061e-01,  5.6624e+00, -6.9458e-02, -6.0412e-01,
          2.5071e+00,  2.9123e-01,  1.7252e+00, -9.6546e+00, -1.2415e+00,
         -1.9517e+00,  2.4769e+00,  5.5917e+00,  2.8032e+00,  2.3195e+01]]
    
    py_posterior_cov = hcat(py_posterior_cov...)
    return py_posterior_cov
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
    n_steps::Int=100, lr::Real=1e-1,
    λinit::Union{Nothing,Real}=nothing,
    σinit::Union{Nothing,Real}=nothing,
    verbose::Bool=false,
    tune_σ::Bool=la.likelihood==:regression
)

    # Setup:
    logP₀ = isnothing(λinit) ? log.(unique(diag(la.P₀))) : log.([λinit])   # prior precision (scalar)
    logσ = isnothing(σinit) ? log.([la.σ]) : log.([σinit])                 # noise (scalar)
    opt = Adam(lr)
    show_every = round(n_steps/10)
    i = 0
    if tune_σ
        @assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
        ps = Flux.params(logP₀,logσ)
    else
        if la.likelihood == :regression
            @warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
        end
        ps = Flux.params(logP₀)
    end
    loss(P₀,σ) = - log_marginal_likelihood(la; P₀=P₀[1], σ=σ[1])

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
