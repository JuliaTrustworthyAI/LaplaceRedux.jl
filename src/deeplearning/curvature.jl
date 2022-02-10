module Curvature

using Flux

abstract type CurvatureInterface end
struct EFInterface <: CurvatureInterface
    model::Any
    𝚯::AbstractArray
end

function full(∇∇::EFInterface; d)
    nn = ∇∇.model
    x, y = d
    # Output:
    ŷ = nn(x)
    # Jacobian:
    𝐉 = jacobian(() -> m(x),params(nn))
    𝐉 = reduce(hcat,[𝐉[θ] for θ ∈ EFInterface.𝚯])
    # Hessian approximation:
    𝐠 = gradient(() -> loss(x,y), params(nn)) 
    𝐠 = reduce(vcat,[vec(𝐠[θ]) for θ ∈ EFInterface.𝚯])
    𝐇 = 𝐠 * 𝐠'
    return ŷ, 𝐉, 𝐇
end


end