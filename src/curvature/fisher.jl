"Constructor for curvature approximated by empirical Fisher."
mutable struct EmpiricalFisher <: CurvatureInterface
    model::Any
    likelihood::Symbol
    loss_fun::Function
    param_indices::Vector{Int}
    factor::Union{Nothing,Real}
    subset_of_weights::Symbol
    subnetwork_indices::Union{Nothing,Vector{Int}}
end

function EmpiricalFisher(
    model::Any,
    likelihood::Symbol,
    param_indices::Vector{Int},
    subset_of_weights::Symbol,
    subnetwork_indices::Union{Nothing,Vector{Int}},
)
    # Define loss function:
    loss_fun = get_loss_fun(likelihood, model)
    factor = likelihood == :regression ? 0.5 : 1.0

    return EmpiricalFisher(
        model, likelihood, loss_fun, param_indices, factor, subset_of_weights, subnetwork_indices
    )
end

"""
    full_unbatched(curvature::EmpiricalFisher, d::Tuple)

Compute the full empirical Fisher for a single datapoint.
"""
function full_unbatched(curvature::EmpiricalFisher, d::Tuple)
    x, y = d

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)
    # Get gradient as a flat vector (already subset-selected)
    𝐠 = gradients(curvature, x, y)

    if curvature.subset_of_weights == :subnetwork
        𝐠 = 𝐠[curvature.subnetwork_indices]
    end

    # Empirical Fisher:
    # - the product of the gradient vector with itself transposed
    H = 𝐠 * 𝐠'

    return loss, H
end

"""
    full_batched(curvature::EmpiricalFisher, d::Tuple)

Compute the full empirical Fisher for batch of inputs-outputs, with the batch dimension at the end.
"""
function full_batched(curvature::EmpiricalFisher, d::Tuple)
    x, y = d

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)
    # Jacobian of per-sample losses via destructure:
    θ, re = Flux.destructure(nn)
    grads_mat = jacobian(
        θ_ -> curvature.loss_fun(re(θ_)(x), y; agg=identity), θ
    )[1]
    𝐠 = transpose(grads_mat[:, curvature.param_indices])
    if curvature.subset_of_weights == :subnetwork
        𝐠 = 𝐠[curvature.subnetwork_indices, :]
    end

    # Empirical Fisher:
    # H = 𝐠 * 𝐠'
    @tullio H[i, j] := 𝐠[i, b] * 𝐠[j, b]

    return loss, H
end
