"""
    Posterior

Container for the results of a Laplace approximation.

# Fields

- `μ::AbstractVector`: the MAP estimate of the parameters
- `H::Union{AbstractArray,AbstractDecomposition,Nothing}`: the Hessian matrix
- `P::Union{AbstractArray,AbstractDecomposition,Nothing}`: the posterior precision matrix
- `Σ::Union{AbstractArray,Nothing}`: the posterior covariance matrix
- `n_data::Union{Int,Nothing}`: the number of data points
- `n_params::Union{Int,Nothing}`: the number of parameters
- `n_out::Union{Int,Nothing}`: the number of outputs
- `loss::Real`: the loss value
"""
mutable struct Posterior
    μ::AbstractVector
    H::Union{AbstractArray,AbstractDecomposition,Nothing}
    P::Union{AbstractArray,AbstractDecomposition,Nothing}
    Σ::Union{AbstractArray,Nothing}
    n_data::Union{Int,Nothing}
    n_params::Union{Int,Nothing}
    n_out::Union{Int,Nothing}
    loss::Real
end

"""
    Posterior(model::Any, est_params::EstimationParams)

Outer constructor for `Posterior` object.
"""
function Posterior(model::Any, est_params::EstimationParams)
    return Posterior(
        get_map_estimate(model, est_params),
        nothing,
        nothing,
        nothing,
        nothing,
        n_params(model, est_params),
        outdim(model),
        0.0,
    )
end

function Base.:(==)(a::Posterior, b::Posterior) 
    checks = [getfield(a, x)==getfield(b, x) for x in fieldnames(typeof(a))] 
    return all(checks)
end
