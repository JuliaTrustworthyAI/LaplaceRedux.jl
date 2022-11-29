using .Curvature
using Flux
using Flux.Optimise: Adam, update!
using Flux.Optimisers: destructure
using LinearAlgebra
using MLJFlux

mutable struct Laplace <: BaseLaplace
    model::Union{Flux.Chain,MLJFlux.MLJFluxModel}
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
    functional_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.

"""
function functional_variance(la::Laplace,𝐉)
    Σ = posterior_covariance(la)
    fvar = map(j -> (j' * Σ * j), eachcol(𝐉))
    return fvar
end


