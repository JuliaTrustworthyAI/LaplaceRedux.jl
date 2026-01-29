"Constructor for curvature approximated by empirical Fisher."
mutable struct EmpiricalFisher <: CurvatureInterface
    model::Any
    likelihood::Symbol
    loss_fun::Function
    factor::Union{Nothing,Real}
    subset_of_weights::Symbol
    subnetwork_indices::Union{Nothing,Vector{Int}}
end

function EmpiricalFisher(
    model::Any,
    likelihood::Symbol,
    subset_of_weights::Symbol,
    subnetwork_indices::Union{Nothing,Vector{Int}},
)
    # Define loss function:
    loss_fun = get_loss_fun(likelihood, model)
    factor = likelihood == :regression ? 0.5 : 1.0

    return EmpiricalFisher(
        model, likelihood, loss_fun, factor, subset_of_weights, subnetwork_indices
    )
end

"""
    full_unbatched(curvature::EmpiricalFisher, d::Tuple)

Compute the full empirical Fisher for a single datapoint.
"""
function full_unbatched(curvature::EmpiricalFisher, d::Tuple)
    x, y = d
    nn = curvature.model
    
    # Compute loss
    loss = curvature.factor * curvature.loss_fun(nn(x), y)
    
    # Use destructure to get flat parameters
    p_flat, restructure = Flux.destructure(nn)
    
    # Compute gradient with respect to flat parameters
    grads = Flux.gradient(p_flat) do p
        m = restructure(p)
        curvature.loss_fun(m(x), y)
    end
    
    # Extract gradient vector
    𝐠 = grads[1]
    
    # Apply subnetwork masking if needed
    if curvature.subset_of_weights == :subnetwork
        𝐠 = 𝐠[curvature.subnetwork_indices]
    end

    # Empirical Fisher: outer product of gradient with itself
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
    𝐠 = jac[1]
    
    # Reshape to [n_params, n_samples]
    if ndims(𝐠) == 3
        # Multiple outputs: flatten output dimension with batch dimension
        n_out, n_params, batch_size = size(𝐠)
        𝐠 = reshape(permutedims(𝐠, (2, 1, 3)), n_params, :)
    elseif ndims(𝐠) == 2
        # Single output: already correct shape or needs transpose
        if size(𝐠, 1) != length(p_flat)
            𝐠 = transpose(𝐠)
        end
    end
    
    # Apply subnetwork masking if needed
    if curvature.subset_of_weights == :subnetwork
        𝐠 = 𝐠[curvature.subnetwork_indices, :]
    end

    # Empirical Fisher: H = 𝐠 * 𝐠'
    @tullio H[i, j] := 𝐠[i, b] * 𝐠[j, b]

    return loss, H
end
