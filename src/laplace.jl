using .Curvature: Kron, KronDecomposed, mm
using Flux
using Flux.Optimise: Adam, update!
using Flux.Optimisers: destructure
using LinearAlgebra
using MLUtils

"""
Compile-time copy-paste macro @def: a macro that creates a macro with the specified name and content,
which is then immediately applied to the code.

Ref: https://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/
"""
macro def(name, definition)
    return quote
        macro $(esc(name))()
            return esc($(Expr(:quote, definition)))
        end
    end
end

@def fields_baselaplace begin
    model::Flux.Chain
    likelihood::Symbol
    subset_of_weights::Symbol
    # indices of the subnetwork
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}
    hessian_structure::Symbol
    curvature::Union{Curvature.CurvatureInterface,Nothing}
    # standard deviation in the Gaussian prior
    σ::Real
    # prior mean
    μ₀::Real
    # posterior mean
    μ::AbstractVector
    # prior precision (i.e. inverse covariance matrix)
    P₀::Union{AbstractMatrix,UniformScaling}
    # Hessian matrix
    H::Union{AbstractArray,KronDecomposed,Nothing}
    # posterior precision
    P::Union{AbstractArray,KronDecomposed,Nothing}
    # posterior covariance matrix
    Σ::Union{AbstractArray,Nothing}
    n_params::Union{Int,Nothing}
    n_data::Union{Int,Nothing}
    n_out::Union{Int,Nothing}
    loss::Real
end

mutable struct Laplace <: BaseLaplace
    # NOTE: following the advice of Chr. Rackauckas, common BaseLaplace fields are inherited via macros, zero-cost
    # Ref: https://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/
    @fields_baselaplace
end

mutable struct KronLaplace <: BaseLaplace
    @fields_baselaplace
end

using Parameters

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

"""
validate_subnetwork_indices(
subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}, params
)

Determines whether subnetwork_indices is a valid input for specified parameters.
"""
function validate_subnetwork_indices(
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}, params
)
    @assert (subnetwork_indices !== nothing) "If `subset_of_weights` is `:subnetwork`, then `subnetwork_indices` should be a vector of vectors of integers."
    # Check if subnetwork_indices is a vector containing an empty vector
    @assert !(subnetwork_indices == [[]]) "If `subset_of_weights` is `:subnetwork`, then `subnetwork_indices` should be a vector of vectors of integers."
    # Initialise a set of vectors
    selected = Set{Vector{Int}}()
    for (i, index) in enumerate(subnetwork_indices)
        @assert !(index in selected) "Element $(i) in `subnetwork_indices` should be unique."
        theta_index = index[1]
        @assert (theta_index in 1:length(params)) "The first index of element $(i) in `subnetwork_indices` should be between 1 and $(length(params))."
        # Calculate number of dimensions of a parameter
        theta_dims = size(params[theta_index])
        @assert length(index) - 1 == length(theta_dims) "Element $(i) in `subnetwork_indices` should have $(theta_dims) coordinates."
        for j in eachindex(index)[2:end]
            @assert (index[j] in 1:theta_dims[j - 1]) "The index $(j) of element $(i) in `subnetwork_indices` should be between 1 and $(theta_dims[j - 1])."
        end
        push!(selected, index)
    end
end

"""
convert_subnetwork_indices(subnetwork_indices::AbstractArray)

Converts the subnetwork indices from the user given format [theta, row, column] to an Int i that corresponds to the index
of that weight in the flattened array of weights.
"""
function convert_subnetwork_indices(
    subnetwork_indices::Vector{Vector{Int}}, params::AbstractArray
)
    converted_indices = Vector{Int}()
    for i in subnetwork_indices
        flat_theta_index = reduce((acc, p) -> acc + length(p), params[1:(i[1] - 1)]; init=0)
        if length(i) == 2
            push!(converted_indices, flat_theta_index + i[2])
        elseif length(i) == 3
            push!(
                converted_indices,
                flat_theta_index + (i[2] - 1) * size(params[i[1]], 2) + i[3],
            )
        end
    end
    return converted_indices
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

function _fit!(
    la::KronLaplace, data; batched::Bool=false, batchsize::Int, override::Bool=true
)
    @assert !batched "Batched Kronecker-factored Laplace approximations not supported"
    @assert la.likelihood == :classification &&
        get_loss_type(la.likelihood, la.curvature.model) == :logitcrossentropy "Only multi-class classification supported"

    # NOTE: the fitting process is structured differently for Kronecker-factored methods
    # to avoid allocation, initialisation & interleaving overhead
    # Thus the loss, Hessian, and data size is computed not in a loop but in a separate function.
    loss, H, n_data = Curvature.kron(la.curvature, data; batched=batched)

    la.loss = loss
    la.H = H
    la.P = posterior_precision(la)
    # NOTE: like in laplace-torch, post covariance is not defined for KronLaplace
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

"""
functional_variance(la::KronLaplace, 𝐉::Matrix)

Compute functional variance for the GLM predictive: as the diagonal of the K×K predictive output covariance matrix 𝐉𝐏⁻¹𝐉ᵀ,
where K is the number of outputs, 𝐏 is the posterior precision, and 𝐉 is the Jacobian of model output `𝐉=∇f(x;θ)|θ̂`.
"""
function functional_variance(la::KronLaplace, 𝐉::Matrix)
    return diag(inv_square_form(la.P, 𝐉))
end

"""
function inv_square_form(K::KronDecomposed, W::Matrix)

Special function to compute the inverse square form 𝐉𝐏⁻¹𝐉ᵀ (or 𝐖𝐊⁻¹𝐖ᵀ)
"""
function inv_square_form(K::KronDecomposed, W::Matrix)
    SW = mm(K, W; exponent=-1)
    return W * SW'
end
