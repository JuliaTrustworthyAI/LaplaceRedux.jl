"""
    Prior

Container for the prior parameters of a Laplace approximation.

# Fields

- `observational_noise::Real`: the observation noise
- `prior_mean::Real`: the prior mean
- `prior_precision::Real`: the prior precision
- `prior_precision_matrix::Union{Nothing,AbstractMatrix,UniformScaling}`: the prior precision matrix
"""
mutable struct Prior
    observational_noise::Real
    prior_mean::Real
    prior_precision::Real
    prior_precision_matrix::Union{Nothing,AbstractMatrix,UniformScaling}
end

function Base.getproperty(ce::Prior, sym::Symbol)
    sym = sym === :σ ? :observational_noise : sym
    sym = sym === :μ₀ ? :prior_mean : sym
    sym = sym === :λ ? :prior_precision : sym
    sym = sym === :P₀ ? :prior_precision_matrix : sym
    return Base.getfield(ce, sym)
end

function Base.setproperty!(ce::Prior, sym::Symbol, val)
    sym = sym === :σ ? :observational_noise : sym
    sym = sym === :μ₀ ? :prior_mean : sym
    sym = sym === :λ ? :prior_precision : sym
    sym = sym === :P₀ ? :prior_precision_matrix : sym
    return Base.setfield!(ce, sym, val)
end

"""
    Prior(params::LaplaceParams)

Extracts the prior parameters from a `LaplaceParams` object.
"""
function Prior(params::LaplaceParams, model::Any, likelihood::Symbol)
    prior_precision_matrix = params.prior_precision_matrix
    n = LaplaceRedux.n_params(model, EstimationParams(params, model, likelihood))
    if typeof(prior_precision_matrix) <: UniformScaling
        prior_precision_matrix = prior_precision_matrix(n)
    elseif isnothing(prior_precision_matrix)
        prior_precision_matrix = UniformScaling(params.prior_precision)(n)
    end
    # Sanity:
    if isa(prior_precision_matrix, AbstractMatrix)
        @assert all(size(prior_precision_matrix) .== n) "Dimensions of prior Hessian $(size(prior_precision_matrix)) do not align with number of parameters ($n)"
    end
    return Prior(params.observational_noise, params.prior_mean, params.prior_precision, prior_precision_matrix)
end
