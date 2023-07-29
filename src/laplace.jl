using .Curvature: Kron, KronDecomposed, mm
using Flux
using Flux.Optimise: Adam, update!
using Flux.Optimisers: destructure
using LinearAlgebra
using MLUtils
using Parameters

mutable struct Laplace <: BaseLaplace
    # NOTE: following the advice of Chr. Rackauckas, common BaseLaplace fields are inherited via macros, zero-cost
    # Ref: https://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/
    @fields_baselaplace
end

@with_kw struct LaplaceParams
    subset_of_weights::Symbol = :all
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}} = nothing
    hessian_structure::Symbol = :full
    backend::Symbol = :GGN
    σ::Real = 1.0
    μ₀::Real = 0.0
    # regularization parameter
    λ::Real = 1.0
    P₀::Union{Nothing,AbstractMatrix,UniformScaling} = nothing
    loss::Real = 0.0
end

"""
Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)

Wrapper function to prepare Laplace approximation.
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...)

    # Load hyperparameters:
    args = LaplaceParams(; kwargs...)

    # Assertions:
    @assert !(args.σ != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."
    @assert args.subset_of_weights ∈ [:all, :last_layer, :subnetwork] "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer, :subnetwork]`"
    if (args.subset_of_weights == :subnetwork)
        validate_subnetwork_indices(args.subnetwork_indices, Flux.params(model))
    end

    # Setup:
    P₀ = isnothing(args.P₀) ? UniformScaling(args.λ) : args.P₀
    nn = model
    n_out = outdim(nn)
    μ = reduce(vcat, [vec(θ) for θ in Flux.params(nn)])                       # μ contains the vertically concatenated parameters of the neural network

    # Concrete subclass constructor
    # NOTE: Laplace is synonymous to FullLaplace
    constructor = args.hessian_structure == :kron ? KronLaplace : Laplace

    # TODO: this may be cleaner with Base.@kwdef
    # Instantiate LA:
    la = constructor(
        model,
        likelihood,
        args.subset_of_weights,
        args.subnetwork_indices,
        args.hessian_structure,
        nothing,
        args.σ,
        args.μ₀,
        μ,
        P₀,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        n_out,
        args.loss,
    )

    params = get_params(la)

    # Instantiating curvature interface
    subnetwork_indices = if la.subset_of_weights == :subnetwork
        convert_subnetwork_indices(la.subnetwork_indices, params)
    else
        nothing
    end
    la.curvature = getfield(Curvature, args.backend)(
        nn, likelihood, params, la.subset_of_weights, subnetwork_indices
    )

    if la.subset_of_weights == :subnetwork
        la.n_params = length(la.subnetwork_indices)
    else
        la.n_params = length(reduce(vcat, [vec(θ) for θ in params]))                # number of params
    end
    la.μ = la.μ[(end - la.n_params + 1):end]                                    # adjust weight vector
    if typeof(la.P₀) <: UniformScaling
        la.P₀ = la.P₀(la.n_params)
    end

    # Sanity:
    if isa(la.P₀, AbstractMatrix)
        @assert all(size(la.P₀) .== la.n_params) "Dimensions of prior Hessian $(size(la.P₀)) do not align with number of parameters ($(la.n_params))"
    end

    return la
end

function _fit!(la::Laplace, data; batched::Bool=false, batchsize::Int, override::Bool=true)
    if override
        H = _init_H(la)
        loss = 0.0
        n_data = 0
    end

    for d in data
        loss_batch, H_batch = hessian_approximation(la, d; batched=batched)
        loss += loss_batch
        H += H_batch
        n_data += batchsize
    end

    # Store output:
    la.loss = loss
    # Hessian
    la.H = H
    # Posterior precision
    la.P = posterior_precision(la)
    # Posterior covariance
    la.Σ = posterior_covariance(la, la.P)
    la.curvature.params = get_params(la)
    # Number of observations
    return la.n_data = n_data
end

"""
functional_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.
"""
function functional_variance(la::Laplace, 𝐉)
    Σ = posterior_covariance(la)
    fvar = map(j -> (j' * Σ * j), eachrow(𝐉))
    return fvar
end
