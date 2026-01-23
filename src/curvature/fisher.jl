"Constructor for curvature approximated by empirical Fisher."
mutable struct EmpiricalFisher <: CurvatureInterface
    model::Any
    likelihood::Symbol
    loss_fun::Function
    params::AbstractArray
    factor::Union{Nothing,Real}
    subset_of_weights::Symbol
    subnetwork_indices::Union{Nothing,Vector{Int}}
end

function EmpiricalFisher(
    model::Any,
    likelihood::Symbol,
    params::AbstractArray,
    subset_of_weights::Symbol,
    subnetwork_indices::Union{Nothing,Vector{Int}},
)
    # Define loss function:
    loss_fun = get_loss_fun(likelihood, model)
    factor = likelihood == :regression ? 0.5 : 1.0

    return EmpiricalFisher(
        model, likelihood, loss_fun, params, factor, subset_of_weights, subnetwork_indices
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
    𝐠 = gradients(curvature, x, y)
    # Concatenate the selected gradients into a vector, column-wise
    𝐠 = reduce(vcat, [vec(𝐠[θ]) for θ in curvature.params])

    if curvature.subset_of_weights == :subnetwork
        𝐠 = [𝐠[p] for p in curvature.subnetwork_indices]
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
    nn = curvature.model
    grads::Zygote.Grads = jacobian(
        () -> curvature.loss_fun(nn(x), y; agg=identity), Flux.params(nn)
    )
    𝐠 = transpose(reduce(hcat, [grads[θ] for θ in curvature.params]))
    if curvature.subset_of_weights == :subnetwork
        𝐠 = 𝐠[curvature.subnetwork_indices, :]
    end

    # Empirical Fisher:
    # H = 𝐠 * 𝐠'
    @tullio H[i, j] := 𝐠[i, b] * 𝐠[j, b]

    return loss, H
end
