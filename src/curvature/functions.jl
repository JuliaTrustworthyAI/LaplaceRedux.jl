using Flux
using ..LaplaceRedux: get_loss_fun, outdim
using LinearAlgebra
using Zygote

"Basetype for any curvature interface."
abstract type CurvatureInterface end 

"""
    jacobians(curvature::CurvatureInterface, X::AbstractArray)

Computes the Jacobian `∇f(x;θ)` where `f: ℝᴰ ↦ ℝᴷ`.
"""

function transform_jacobians(curvature::CurvatureInterface, J::Zygote.Grads)
    Js = []
    for θ ∈ curvature.params
        param_size = size(θ)
        indices = collect(1:length(vec(θ')))
        updated_indices = vec(reshape(indices, param_size)')
        Jk = J[θ]'
        push!(Js, Jk[updated_indices, :])
    end
    Js = hcat(Js'...)'
    return Js
end


function jacobians(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    # Jacobian:
    𝐉 = jacobian(() -> nn(X),Flux.params(nn))
    # 𝐉 = permutedims(reduce(hcat,[𝐉[θ] for θ ∈ curvature.params]))
    𝐉 = transform_jacobians(curvature, 𝐉)
    return 𝐉, ŷ
end

"""
    gradients(curvature::CurvatureInterface, X::AbstractArray, y::Number)

Compute the gradients with respect to the loss function: `∇ℓ(f(x;θ),y)` where `f: ℝᴰ ↦ ℝᴷ`.
"""
function gradients(curvature::CurvatureInterface, X::AbstractArray, y::Union{Number, AbstractArray})
    model = curvature.model
    𝐠 = gradient(() -> curvature.loss_fun(X,y),Flux.params(model)) 
    return 𝐠
end

# "Constructor for Generalized Gauss Newton."
# struct GGN <: CurvatureInterface
#     model::Any
#     likelihood::Symbol
#     loss_fun::Function
#     params::AbstractArray
#     factor::Union{Nothing,Real}
# end

# function GGN(model::Any, likelihood::Symbol, params::AbstractArray)

#     @error "GGN not yet implemented."

#     # Define loss function:
#     loss_fun = get_loss_fun(likelihood, model)
#     factor = likelihood == :regression ? 0.5 : 1.0

#     GGN(model, likelihood, loss_fun, params, factor)
# end

# """
#     full(curvature::GGN, d::Union{Tuple,NamedTuple})

# Compute the full GGN.
# """
# function full(curvature::GGN, d::Tuple)
#     x, y = d

#     loss = curvature.factor * curvature.loss_fun(x, y)

#     𝐉, fμ = jacobians(curvature, x)

#     if curvature.likelihood == :regression
#         H = 𝐉 * 𝐉'
#     else
#         p = outdim(curvature.model) > 1 ? softmax(fμ) : sigmoid(fμ)
#         H = map(j -> j * (diagm(p) - p * p') * j', eachcol(𝐉))
#         println(H)
#     end
    
#     return loss, H

# end

"Constructor for Empirical Fisher."
struct EmpiricalFisher <: CurvatureInterface
    model::Any
    likelihood::Symbol
    loss_fun::Function
    params::AbstractArray
    factor::Union{Nothing,Real}
end

function EmpiricalFisher(model::Any, likelihood::Symbol, params::AbstractArray)

    # Define loss function:
    loss_fun = get_loss_fun(likelihood, model)
    factor = likelihood == :regression ? 0.5 : 1.0

    EmpiricalFisher(model, likelihood, loss_fun, params, factor)
end

"""
    full(curvature::EmpiricalFisher, d::Union{Tuple,NamedTuple})

Compute the full empirical Fisher.
"""
function full(curvature::EmpiricalFisher, d::Tuple)
    x, y = d

    loss = curvature.factor * curvature.loss_fun(x, y)
    𝐠 = gradients(curvature, x, y) 
    𝐠 = reduce(vcat,[vec(𝐠[θ]) for θ ∈ curvature.params])

    # Empirical Fisher:
    H = 𝐠 * 𝐠'
    
    return loss, H

end