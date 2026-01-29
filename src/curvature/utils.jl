using ChainRulesCore
using Flux

"""
    jacobians(curvature::CurvatureInterface, X::AbstractArray; batched::Bool=false)

Computes the Jacobian ∇f(x;θ) where f: ℝᴰ ↦ ℝᴷ.
"""
function jacobians(curvature::CurvatureInterface, X::AbstractArray; batched::Bool=false)
    if batched
        return jacobians_batched(curvature, X)
    else
        return jacobians_unbatched(curvature, X)
    end
end

"""
    jacobians_unbatched(curvature::CurvatureInterface, X::AbstractArray)

Compute the Jacobian of the model output w.r.t. model parameters for the point X, without batching.
"""
function jacobians_unbatched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    
    # Use destructure for flat parameter handling
    p_flat, restructure = Flux.destructure(nn)
    
    # Output
    y_hat = nn(X)
    y_hat = vec(y_hat)
    
    # Jacobian: differentiate f with regards to the model parameters
    J = []
    ChainRulesCore.ignore_derivatives() do
        jac = Zygote.jacobian(p_flat) do p
            m = restructure(p)
            vec(m(X))
        end
        push!(J, jac[1])  # Extract the Jacobian from the tuple
    end
    jac_matrix = J[1]
    
    # jac_matrix is now [K, P] where K is output size, P is number of parameters
    # Apply subnetwork masking if needed
    if curvature.subset_of_weights == :subnetwork
        jac_matrix = jac_matrix[:, curvature.subnetwork_indices]
    end
    
    return jac_matrix, y_hat
end

"""
    jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)

Compute Jacobians of the model output w.r.t. model parameters for points in X, with batching.
"""
function jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    
    # Use destructure for flat parameter handling
    p_flat, restructure = Flux.destructure(nn)
    
    # Output
    y_hat = nn(X)
    batch_size = size(X)[end]
    out_size = outdim(nn)
    
    # Jacobian
    grads = []
    ChainRulesCore.ignore_derivatives() do
        g = Zygote.jacobian(p_flat) do p
            m = restructure(p)
            m(X)
        end
        push!(grads, g[1])  # Extract the Jacobian from the tuple
    end
    jac_raw = grads[1]
    
    # jac_raw shape depends on output structure
    # Typically: [out_size * batch_size, n_params] or [out_size, n_params, batch_size]
    if ndims(jac_raw) == 3
        # Shape: [out_size, n_params, batch_size]
        jac_matrix = permutedims(jac_raw, (1, 2, 3))
    elseif ndims(jac_raw) == 2
        # Shape: [out_size * batch_size, n_params]
        # Reshape to [out_size, batch_size, n_params] then permute
        jac_matrix = reshape(jac_raw, out_size, batch_size, :)
        jac_matrix = permutedims(jac_matrix, (1, 3, 2))  # [out_size, n_params, batch_size]
    end
    
    # Apply subnetwork masking if needed
    if curvature.subset_of_weights == :subnetwork
        jac_matrix = jac_matrix[:, curvature.subnetwork_indices, :]
    end
    
    return jac_matrix, y_hat
end

"""
    gradients(curvature::CurvatureInterface, X::AbstractArray, y::Union{Number,AbstractArray})

Compute the gradients with respect to the loss function: ∇ℓ(f(x;θ),y) where f: ℝᴰ ↦ ℝᴷ.
"""
function gradients(
    curvature::CurvatureInterface, X::AbstractArray, y::Union{Number,AbstractArray}
)
    nn = curvature.model
    g = Flux.gradient(m -> curvature.loss_fun(m(X), y), nn)
    return g[1]
end
