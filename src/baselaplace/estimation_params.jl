"""
    EstimationParams

Container for the parameters of a Laplace approximation. 

# Fields

- `subset_of_weights::Symbol`: the subset of weights to consider. Possible values are `:all`, `:last_layer`, and `:subnetwork`.
- `subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}`: the indices of the subnetwork. Possible values are `nothing` or a vector of vectors of integers.
- `hessian_structure::HessianStructure`: the structure of the Hessian. Possible values are `:full` and `:kron` or a concrete subtype of `HessianStructure`.
- `curvature::Union{Curvature.CurvatureInterface,Nothing}`: the curvature interface. Possible values are `nothing` or a concrete subtype of `CurvatureInterface`.
"""
mutable struct EstimationParams
    subset_of_weights::Symbol
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}
    hessian_structure::HessianStructure
    curvature::Union{Curvature.CurvatureInterface,Nothing}
end

"""
    EstimationParams(params::LaplaceParams)

Extracts the estimation parameters from a `LaplaceParams` object.
"""
function EstimationParams(params::LaplaceParams, model::Any, likelihood::Symbol)

    # Hessian structure:
    hessian_structure = params.hessian_structure
    if !(typeof(hessian_structure) <: HessianStructure)
        hessian_structure =
            String(hessian_structure) == "full" ? FullHessian() : KronHessian()
    end

    # Asserts:
    @assert params.subset_of_weights ∈ [:all, :last_layer, :subnetwork] "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer, :subnetwork]`"
    if (params.subset_of_weights == :subnetwork)
        validate_subnetwork_indices(params.subnetwork_indices, Flux.params(model))
    end

    est_params = EstimationParams(
        params.subset_of_weights,
        params.subnetwork_indices,
        hessian_structure,
        params.curvature,
    )

    # Instantiating curvature interface
    instantiate_curvature!(est_params, model, likelihood, params.backend)

    return est_params
end

"""
    Flux.params(model::Any, params::EstimationParams)

Extracts the parameters of a model based on the subset of weights specified in the `EstimationParams` object.
"""
function Flux.params(model::Any, params::EstimationParams)
    model_params = Flux.params(model)
    n_elements = length(model_params)
    if params.subset_of_weights == :all || params.subset_of_weights == :subnetwork
        # get all parameters and constants in logitbinarycrossentropy
        model_params = [θ for θ in model_params]
    elseif params.subset_of_weights == :last_layer
        # Only get last layer parameters:
        # params[n_elements] is the bias vector of the last layer
        # params[n_elements-1] is the weight matrix of the last layer
        model_params = [model_params[n_elements - 1], model_params[n_elements]]
    end
    return model_params
end

"""
    n_params(model::Any, params::EstimationParams)

Helper function to determine the number of parameters of a `Flux.Chain` with Laplace approximation.
"""
function n_params(model::Any, est_params::EstimationParams)
    if est_params.subset_of_weights == :subnetwork
        n = length(est_params.subnetwork_indices)
    else
        n = length(reduce(vcat, [vec(θ) for θ in Flux.params(model, est_params)]))
    end
    return n
end

"""
    get_map_estimate(model::Any, est_params::EstimationParams)

Helper function to extract the MAP estimate of the parameters for the model based on the subset of weights specified in the `EstimationParams` object.
"""
function get_map_estimate(model::Any, est_params::EstimationParams)
    μ = reduce(vcat, [vec(θ) for θ in Flux.params(model, est_params)])
    return μ[(end - LaplaceRedux.n_params(model, est_params) + 1):end]
end

"""
    instantiate_curvature!(params::EstimationParams, model::Any, likelihood::Symbol, backend::Symbol)

Instantiates the curvature interface for a Laplace approximation. The curvature interface is a concrete subtype of [`CurvatureInterface`](@ref) and is used to compute the Hessian matrix. The curvature interface is stored in the `curvature` field of the `EstimationParams` object.
"""
function instantiate_curvature!(
    params::EstimationParams, model::Any, likelihood::Symbol, backend::Symbol
)
    model_params = Flux.params(model, params)

    if params.subset_of_weights == :subnetwork
        subnetwork_indices = convert_subnetwork_indices(
            params.subnetwork_indices, model_params
        )
    else
        subnetwork_indices = nothing
    end

    curvature = getfield(Curvature, backend)(
        model, likelihood, model_params, params.subset_of_weights, subnetwork_indices
    )

    return params.curvature = curvature
end

function Base.:(==)(a::EstimationParams, b::EstimationParams)
    checks = [getfield(a, x) == getfield(b, x) for x in fieldnames(typeof(a))]
    println(checks)
    return all(checks)
end
