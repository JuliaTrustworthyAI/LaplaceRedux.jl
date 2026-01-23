using ChainRulesCore
using Flux

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
Here, the nn function is wrapped in an anonymous function using the () -> syntax, which allows it to be differentiated using automatic differentiation.
"""
function jacobians_unbatched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    # Convert ŷ to a vector
    ŷ = vec(ŷ)
    # Jacobian:
    # Differentiate f with regards to the model parameters
    J = []
    ChainRulesCore.ignore_derivatives() do
        𝐉 = jacobian(() -> nn(X), Flux.params(nn))
        push!(J, 𝐉)
    end
    𝐉 = J[1]
    # Concatenate Jacobians for the selected parameters, to produce a matrix (K, P), where P is the total number of parameter scalars.                      
    𝐉 = reduce(hcat, [𝐉[θ] for θ in curvature.params])
    if curvature.subset_of_weights == :subnetwork
        𝐉 = 𝐉[:, curvature.subnetwork_indices]
    end
    return 𝐉, ŷ
end

"""
    jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)

Compute Jacobians of the model output w.r.t. model parameters for points in X, with batching.
"""
function jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    batch_size = size(X)[end]
    out_size = outdim(nn)
    # Jacobian:
    grads = []
    ChainRulesCore.ignore_derivatives() do
        g = jacobian(() -> nn(X), Flux.params(nn))
        push!(grads, g)
    end
    grads = grads[1]
    grads_joint = reduce(hcat, [grads[θ] for θ in curvature.params])
    views = [
        @view grads_joint[batch_start:(batch_start + out_size - 1), :] for
        batch_start in 1:out_size:(batch_size * out_size)
    ]
    𝐉 = stack(views)
    if curvature.subset_of_weights == :subnetwork
        𝐉 = 𝐉[:, curvature.subnetwork_indices, :]
    end
    # NOTE: it is also possible to select indices at the view stage TODO benchmark and compare
    return 𝐉, ŷ
end

"""
    gradients(curvature::CurvatureInterface, X::AbstractArray, y::Number)

Compute the gradients with respect to the loss function: `∇ℓ(f(x;θ),y)` where `f: ℝᴰ ↦ ℝᴷ`.
"""
function gradients(
    curvature::CurvatureInterface, X::AbstractArray, y::Union{Number,AbstractArray}
)::Zygote.Grads
    nn = curvature.model
    𝐠 = gradient(() -> curvature.loss_fun(nn(X), y), Flux.params(nn))           # compute the gradients of the loss function with respect to the model parameters
    return 𝐠
end
