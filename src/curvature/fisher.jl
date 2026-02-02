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

    # gradients now returns a flat vector directly
    grad_vec = gradients(curvature, x, y)

    # Apply subnetwork masking if needed
    if curvature.subset_of_weights == :subnetwork
        grad_vec = grad_vec[curvature.subnetwork_indices]
    end

    # Empirical Fisher:
    # - the product of the gradient vector with itself transposed
    H = grad_vec * grad_vec'

    return loss, H
end

"""
    full_batched(curvature::EmpiricalFisher, d::Tuple)

Compute the full empirical Fisher for batch of inputs-outputs, with the batch dimension at the end.
"""
function full_batched(curvature::EmpiricalFisher, d::Tuple)
    x, y = d
    nn = curvature.model

    # Compute loss
    loss = curvature.factor * sum(curvature.loss_fun(nn(x), y; agg=identity))

    # Use destructure for flat parameter handling
    p_flat, restructure = Flux.destructure(nn)

    # Compute Jacobian with respect to flat parameters
    # This gives us per-sample gradients
    jac = Zygote.jacobian(p_flat) do p
        m = restructure(p)
        curvature.loss_fun(m(x), y; agg=identity)
    end

    # jac[1] shape: [n_outputs, n_params, batch_size] or [n_params, batch_size]
    grad_matrix = jac[1]

    # Reshape to [n_params, n_samples]
    if ndims(grad_matrix) == 3
        # Multiple outputs: flatten output dimension with batch dimension
        n_out, n_params, batch_size = size(grad_matrix)
        grad_matrix = reshape(permutedims(grad_matrix, (2, 1, 3)), n_params, :)
    elseif ndims(grad_matrix) == 2
        # Single output: might need transpose to get [n_params, batch_size]
        if size(grad_matrix, 1) != length(p_flat)
            grad_matrix = transpose(grad_matrix)
        end
    end

    # Apply subnetwork masking if needed
    if curvature.subset_of_weights == :subnetwork
        grad_matrix = grad_matrix[curvature.subnetwork_indices, :]
    end

    # Empirical Fisher: H = grad_matrix * grad_matrix'
    @tullio H[i, j] := grad_matrix[i, b] * grad_matrix[j, b]

    return loss, H
end
