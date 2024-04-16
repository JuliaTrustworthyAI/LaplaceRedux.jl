"""
    Prior

Container for the prior parameters of a Laplace approximation.

# Fields

- `σ::Real`: the observation noise
- `μ₀::Real`: the prior mean
- `λ::Real`: the prior precision
- `P₀::Union{Nothing,AbstractMatrix,UniformScaling}`: the prior precision matrix
"""
mutable struct Prior
    σ::Real
    μ₀::Real
    λ::Real
    P₀::Union{Nothing,AbstractMatrix,UniformScaling}
end

"""
    Prior(params::LaplaceParams)

Extracts the prior parameters from a `LaplaceParams` object.
"""
function Prior(params::LaplaceParams, model::Any, likelihood::Symbol)
    P₀ = params.P₀
    n = LaplaceRedux.n_params(model, EstimationParams(params, model, likelihood))
    if typeof(P₀) <: UniformScaling
        P₀ = P₀(n)
    elseif isnothing(P₀)
        P₀ = UniformScaling(params.λ)(n)
    end
    # Sanity:
    if isa(P₀, AbstractMatrix)
        @assert all(size(P₀) .== n) "Dimensions of prior Hessian $(size(P₀)) do not align with number of parameters ($n)"
    end
    return Prior(params.σ, params.μ₀, params.λ, P₀)
end
