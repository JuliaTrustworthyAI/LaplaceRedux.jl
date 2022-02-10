module Curvature

using Flux, Zygote
abstract type CurvatureInterface end 

function jacobians(𝑪::CurvatureInterface, X::AbstractArray)
    nn = 𝑪.model
    # Output:
    ŷ = nn(X)
    # Jacobian:
    𝐉 = jacobian(() -> nn(X),Flux.params(nn))
    𝐉 = reduce(hcat,[𝐉[θ] for θ ∈ 𝑪.𝚯])
    return 𝐉, ŷ
end
struct EFInterface <: CurvatureInterface
    model::Any
    loss::Function
    𝚯::AbstractArray
end

function full(𝑪::EFInterface, d::Tuple)
    nn = 𝑪.model
    x, y = d
    # Hessian approximation:
    𝐠 = gradient(() -> 𝑪. loss(x,y),Flux.params(nn)) 
    𝐠 = reduce(vcat,[vec(𝐠[θ]) for θ ∈ 𝑪.𝚯])
    𝐇 = 𝐠 * 𝐠'
    return 𝐇
end

end