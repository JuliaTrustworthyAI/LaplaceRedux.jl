using Flux, Zygote
abstract type CurvatureInterface end 

"""
    jacobians(curvature::CurvatureInterface, X::AbstractArray)

Computes the Jacobian `∇f(x;θ)` where `f: ℝᴰ ↦ ℝᴷ`.
"""
function jacobians(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    # Jacobian:
    𝐉 = jacobian(() -> nn(X),Flux.params(nn))
    𝐉 = reduce(hcat,[𝐉[θ] for θ ∈ curvature.params])
    return 𝐉, ŷ
end

"""
    gradients(curvature::CurvatureInterface, X::AbstractArray, y::Number)

Compute the gradients with respect to the loss function: `∇ℓ(f(x;θ),y)` where `f: ℝᴰ ↦ ℝᴷ`.
"""
function gradients(curvature::CurvatureInterface, X::AbstractArray, y::Union{Number, AbstractArray})
    model = curvature.model
    𝐠 = gradient(() -> curvature.loss(X,y),Flux.params(model)) 
    return 𝐠
end
struct EmpiricalFisher <: CurvatureInterface
    model::Any
    loss::Function
    params::AbstractArray
end

"""
    full(curvature::EmpiricalFisher, d::Union{Tuple,NamedTuple})

Compute the full empirical Fisher.
"""
function full(curvature::EmpiricalFisher, d::Tuple)
    x, y = d
    𝐠 = gradients(curvature, x, y) 
    𝐠 = reduce(vcat,[vec(𝐠[θ]) for θ ∈ curvature.params])

    # Empirical Fisher:
    H = 𝐠 * 𝐠'
    return H

end