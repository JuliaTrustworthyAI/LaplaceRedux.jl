using ChainRulesCore

"""
    jacobians(curvature::CurvatureInterface, X::AbstractArray; batched::Bool=false)

Computes the Jacobian `∇f(x;θ)` where `f: ℝᴰ ↦ ℝᴷ`.
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
Uses `Flux.destructure` to obtain a flat parameter vector and computes the Jacobian via Zygote.
"""
function jacobians_unbatched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    # Convert ŷ to a vector
    ŷ = vec(ŷ)
    # Jacobian via destructure:
    J_result = []
    ChainRulesCore.ignore_derivatives() do
        θ, re = Flux.destructure(nn)
        𝐉 = jacobian(θ_ -> vec(re(θ_)(X)), θ)[1]
        push!(J_result, 𝐉)
    end
    𝐉 = J_result[1]
    # Select the relevant parameter columns
    𝐉 = 𝐉[:, curvature.param_indices]
    if curvature.subset_of_weights == :subnetwork
        𝐉 = 𝐉[:, curvature.subnetwork_indices]
    end
    return 𝐉, ŷ
end

"""
    jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)

Compute Jacobians of the model output w.r.t. model parameters for points in X, with batching.
"""
function jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    batch_size = size(X)[end]
    out_size = outdim(nn)
    # Jacobian via destructure:
    J_result = []
    ChainRulesCore.ignore_derivatives() do
        θ, re = Flux.destructure(nn)
        g = jacobian(θ_ -> vec(re(θ_)(X)), θ)[1]
        push!(J_result, g)
    end
    grads_joint = J_result[1][:, curvature.param_indices]
    views = [
        @view grads_joint[batch_start:(batch_start + out_size - 1), :] for
        batch_start in 1:out_size:(batch_size * out_size)
    ]
    𝐉 = stack(views)
    if curvature.subset_of_weights == :subnetwork
        𝐉 = 𝐉[:, curvature.subnetwork_indices, :]
    end
    # NOTE: it is also possible to select indices at the view stage TODO benchmark and compare
    return 𝐉, ŷ
end

"""
    gradients(curvature::CurvatureInterface, X::AbstractArray, y)

Compute the gradients with respect to the loss function: `∇ℓ(f(x;θ),y)` where `f: ℝᴰ ↦ ℝᴷ`.
Returns a flat gradient vector for the selected parameter subset.
"""
function gradients(
    curvature::CurvatureInterface, X::AbstractArray, y::Union{Number,AbstractArray}
)
    nn = curvature.model
    θ, re = Flux.destructure(nn)
    𝐠 = Flux.gradient(θ_ -> curvature.loss_fun(re(θ_)(X), y), θ)[1]
    return 𝐠[curvature.param_indices]
end
