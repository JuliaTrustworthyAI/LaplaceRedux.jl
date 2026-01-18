"""
    Posterior

Container for the results of a Laplace approximation.

# Fields

- `posterior_mean::AbstractVector`: the MAP estimate of the parameters
- `H::Union{AbstractArray,AbstractDecomposition,Nothing}`: the Hessian matrix
- `P::Union{AbstractArray,AbstractDecomposition,Nothing}`: the posterior precision matrix
- `posterior_covariance_matrix::Union{AbstractArray,Nothing}`: the posterior covariance matrix
- `n_data::Union{Int,Nothing}`: the number of data points
- `n_params::Union{Int,Nothing}`: the number of parameters
- `n_out::Union{Int,Nothing}`: the number of outputs
- `loss::Real`: the loss value
"""
mutable struct Posterior
    posterior_mean::AbstractVector
    H::Union{AbstractArray,AbstractDecomposition,Nothing}
    P::Union{AbstractArray,AbstractDecomposition,Nothing}
    posterior_covariance_matrix::Union{AbstractArray,Nothing}
    n_data::Union{Int,Nothing}
    n_params::Union{Int,Nothing}
    n_out::Union{Int,Nothing}
    loss::Real
end

function Base.getproperty(ce::Posterior, sym::Symbol)
    sym = sym === :μ ? :posterior_mean : sym
    sym = sym === :Σ ? :posterior_covariance_matrix : sym
    return Base.getfield(ce, sym)
end

function Base.setproperty!(ce::Posterior, sym::Symbol, val)
    sym = sym === :μ ? :posterior_mean : sym
    sym = sym === :Σ ? :posterior_covariance_matrix : sym
    return Base.setfield!(ce, sym, val)
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
