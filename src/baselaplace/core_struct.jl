using Flux: Flux
using LinearAlgebra
using MLUtils

"Abstract base type for all Laplace approximations in this library. All subclasses implemented are parametric."
abstract type AbstractLaplace end

"Abstract type for Hessian structure."
abstract type HessianStructure end

"Concrete type for full Hessian structure. This is the default structure."
struct FullHessian <: HessianStructure end

"Abstract type of Hessian decompositions."
abstract type AbstractDecomposition end

"""
    LaplaceParams

Container for the parameters of a Laplace approximation.

# Fields

- `subset_of_weights::Symbol`: the subset of weights to consider. Possible values are `:all`, `:last_layer`, and `:subnetwork`.
- `subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}`: the indices of the subnetwork. Possible values are `nothing` or a vector of vectors of integers.
- `hessian_structure::HessianStructure`: the structure of the Hessian. Possible values are `:full` and `:kron` or a concrete subtype of `HessianStructure`.
- `backend::Symbol`: the backend to use. Possible values are `:GGN` and `:Fisher`.
- `curvature::Union{Curvature.CurvatureInterface,Nothing}`: the curvature interface. Possible values are `nothing` or a concrete subtype of `CurvatureInterface`.
- `observational_noise::Real`: the observation noise
- `prior_mean::Real`: the prior mean of the network parameters.
- `prio_precision::Real`: the prior precision for the network parameters.
- `prior_precision_matrix::Union{Nothing,AbstractMatrix,UniformScaling}`: the prior precision matrix for the network parameters.
"""
Base.@kwdef struct LaplaceParams
    subset_of_weights::Symbol = :all
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}} = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} = FullHessian()
    backend::Symbol = :GGN
    curvature::Union{Curvature.CurvatureInterface,Nothing} = nothing
    observational_noise::Real = 1.0
    prior_mean::Real = 0.0
    prior_precision::Real = 1.0
    prior_precision_matrix::Union{Nothing,AbstractMatrix,UniformScaling} = nothing
end

function Base.getproperty(ce::LaplaceParams, sym::Symbol)
    sym = sym === :σ ? :observational_noise : sym
    sym = sym === :μ₀ ? :prior_mean : sym
    sym = sym === :λ ? :prior_precision : sym
    sym = sym === :P₀ ? :prior_precision_matrix : sym
    return Base.getfield(ce, sym)
end

function Base.setproperty!(ce::LaplaceParams, sym::Symbol, val)
    sym = sym === :σ ? :observational_noise : sym
    sym = sym === :μ₀ ? :prior_mean : sym
    sym = sym === :λ ? :prior_precision : sym
    sym = sym === :P₀ ? :prior_precision_matrix : sym
    return Base.setfield!(ce, sym, val)
end

include("estimation_params.jl")
include("prior.jl")
include("posterior.jl")

"""
    Laplace

Concrete type for Laplace approximation. This type is a subtype of `AbstractLaplace` and is used to store all the necessary information for a Laplace approximation.

# Fields

- `model::Flux.Chain`: The model to be approximated.
- `likelihood::Symbol`: The likelihood function to be used.
- `est_params::`[`EstimationParams`](@ref): The estimation parameters.
- `prior::`[`Prior`](@ref): The parameters defining prior distribution.
- `posterior::`[`Posterior`](@ref): The posterior distribution.
"""
mutable struct Laplace <: AbstractLaplace
    model::Flux.Chain
    likelihood::Symbol
    est_params::EstimationParams
    prior::Prior
    posterior::Posterior
end

"""
    Laplace(model::Any; likelihood::Symbol, kwargs...)

Outer constructor for Laplace approximation. This function constructs a Laplace object from a given model and likelihood function.

# Arguments

- `model::Any`: The model to be approximated (a Flux.Chain).
- `likelihood::Symbol`: The likelihood function to be used. Possible values are `:regression` and `:classification`.

# Keyword Arguments

See [`LaplaceParams`](@ref) for a description of the keyword arguments.

# Returns

- `la::Laplace`: The Laplace object.

# Examples

```julia
using Flux, LaplaceRedux
nn = Chain(Dense(2,1))
la = Laplace(nn, likelihood=:regression)
```
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...)
    args = LaplaceParams(; kwargs...)
    @assert !(args.observational_noise != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."

    # Unpack arguments and wrap in containers:
    est_args = EstimationParams(args, model, likelihood)
    prior = Prior(args, model, likelihood)
    posterior = Posterior(model, est_args)

    # Instantiate Laplace object:
    la = Laplace(model, likelihood, est_args, prior, posterior)

    return la
end

include("utils.jl")
include("fitting.jl")
include("predicting.jl")
include("optimize_prior.jl")
