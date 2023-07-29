"Constructor for curvature approximated by Generalized Gauss-Newton."
mutable struct GGN <: CurvatureInterface
    model::Any
    likelihood::Symbol
    loss_fun::Function
    params::AbstractArray
    factor::Union{Nothing,Real}
    subset_of_weights::Symbol
    subnetwork_indices::Union{Nothing,Vector{Int}}
end

function GGN(
    model::Any,
    likelihood::Symbol,
    params::AbstractArray,
    subset_of_weights::Symbol,
    subnetwork_indices::Union{Nothing,Vector{Int}},
)
    # Define loss function:
    loss_fun = get_loss_fun(likelihood, model)
    factor = likelihood == :regression ? 0.5 : 1.0

    return GGN(
        model, likelihood, loss_fun, params, factor, subset_of_weights, subnetwork_indices
    )
end

"""
    full_unbatched(curvature::GGN, d::Tuple)

Compute the full GGN for a singular input-ouput datapoint. 
"""
function full_unbatched(curvature::GGN, d::Tuple)
    x, y = d

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)

    𝐉, fμ = jacobians(curvature, x)

    if curvature.likelihood == :regression
        H = 𝐉' * 𝐉
    else
        p = outdim(nn) > 1 ? softmax(fμ) : sigmoid(fμ)
        H_lik = diagm(p) - p * p'
        H = 𝐉' * H_lik * 𝐉
    end

    return loss, H
end

"""
    full_batched(curvature::GGN, d::Tuple)

Compute the full GGN for batch of inputs-outputs, with the batch dimension at the end.
"""
function full_batched(curvature::GGN, d::Tuple)
    x, y = d

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)

    𝐉, fμ = jacobians_batched(curvature, x)

    if curvature.likelihood == :regression
        @tullio H[i, j] := 𝐉[k, i, b] * 𝐉[k, j, b]
    else
        p = outdim(nn) > 1 ? softmax(fμ) : sigmoid(fμ)
        # H_lik = diagm(p) - p * p'
        @tullio H_lik[i, j, b] := -p[i, b] * p[j, b]
        @tullio H_lik[i, i, b] += p[i, b]
        # H = 𝐉 * H_lik * 𝐉'
        @tullio H[i, j] := 𝐉[c, i, b] * H_lik[c, k, b] * 𝐉[k, j, b]
    end

    return loss, H
end
