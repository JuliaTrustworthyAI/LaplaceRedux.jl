using Flux
using ..LaplaceRedux: get_loss_fun, outdim
using LinearAlgebra
using Zygote
using Tullio
using Compat

"Basetype for any curvature interface."
abstract type CurvatureInterface end

"""
    jacobians(curvature::CurvatureInterface, X::AbstractArray)

Computes the Jacobian `∇f(x;θ)` where `f: ℝᴰ ↦ ℝᴷ`.
The Jacobian function can be used to compute the Jacobian of any function that supports automatic differentiation. 
Here, the nn function is wrapped in an anonymous function using the () -> syntax, which allows it to be differentiated using automatic differentiation.
"""

function jacobians(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    # Convert ŷ to a vector
    ŷ = vec(ŷ)
    # Jacobian:
    # Differentiate f with regards to the model parameters
    𝐉 = jacobian(() -> nn(X), Flux.params(nn))
    # Concatenate Jacobians for the selected parameters, to produce a matrix (K, P), where P is the total number of parameter scalars.                      
    𝐉 = reduce(hcat, [𝐉[θ] for θ in curvature.params])
    if curvature.subset_of_weights == :subnetwork
        𝐉 = 𝐉[:, curvature.subnetwork_indices]
    end
    return 𝐉, ŷ
end

function jacobians_batched(curvature::CurvatureInterface, X::AbstractArray)
    nn = curvature.model
    # Output:
    ŷ = nn(X)
    batch_size = size(X)[end]
    out_size = outdim(nn)
    # Jacobian:
    grads = jacobian(() -> nn(X), Flux.params(nn))
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
    model = curvature.model
    𝐠 = gradient(() -> curvature.loss_fun(X, y), Flux.params(model))           # compute the gradients of the loss function with respect to the model parameters
    return 𝐠
end

"Constructor for Empirical Fisher."
mutable struct GGN <: CurvatureInterface
    model::Any
    likelihood::Symbol
    loss_fun::Function
    params::AbstractArray
    factor::Union{Nothing,Real}
    subset_of_weights::Symbol
    subnetwork_indices::Union{Nothing,Vector{Int}}
end

function GGN(
    model::Any,
    likelihood::Symbol,
    params::AbstractArray,
    subset_of_weights::Symbol,
    subnetwork_indices::Union{Nothing,Vector{Int}},
)
    # Define loss function:
    loss_fun = get_loss_fun(likelihood, model)
    factor = likelihood == :regression ? 0.5 : 1.0

    return GGN(
        model, likelihood, loss_fun, params, factor, subset_of_weights, subnetwork_indices
    )
end

"""
    full(curvature::GGN, d::Union{Tuple,NamedTuple})

Compute the full GGN.
"""
function full(curvature::GGN, d::Tuple; batched::Bool=false)
    if batched
        full_batched(curvature, d)
    else
        full_unbatched(curvature, d)
    end
end

"""
    full(curvature::GGN, d::Union{Tuple,NamedTuple})

Compute the full GGN for a singular input-ouput datapoint. 
"""
function full_unbatched(curvature::GGN, d::Tuple)
    x, y = d

    loss = curvature.factor * curvature.loss_fun(x, y)

    𝐉, fμ = jacobians(curvature, x)

    if curvature.likelihood == :regression
        H = 𝐉' * 𝐉
    else
        p = outdim(curvature.model) > 1 ? softmax(fμ) : sigmoid(fμ)
        H_lik = diagm(p) - p * p'
        H = 𝐉' * H_lik * 𝐉
    end

    return loss, H
end

"""
    full(curvature::GGN, d::Union{Tuple,NamedTuple})

Compute the full GGN for batch of inputs-outputs, with the batch dimension at the end.
"""
function full_batched(curvature::GGN, d::Tuple)
    x, y = d

    loss = curvature.factor * curvature.loss_fun(x, y)

    𝐉, fμ = jacobians_batched(curvature, x)

    if curvature.likelihood == :regression
        @tullio H[i, j] := 𝐉[k, i, b] * 𝐉[k, j, b]
    else
        p = outdim(curvature.model) > 1 ? softmax(fμ) : sigmoid(fμ)
        # H_lik = diagm(p) - p * p'
        @tullio H_lik[i, j, b] := -p[i, b] * p[j, b]
        @tullio H_lik[i, i, b] += p[i, b]
        # H = 𝐉 * H_lik * 𝐉'
        @tullio H[i, j] := 𝐉[c, i, b] * H_lik[c, k, b] * 𝐉[k, j, b]
    end

    return loss, H
end

"Constructor for Empirical Fisher."
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
    full(curvature::EmpiricalFisher, d::Union{Tuple,NamedTuple})

Compute the full empirical Fisher for a singular input-ouput datapoint. 
"""
function full(curvature::EmpiricalFisher, d::Tuple; batched::Bool=false)
    if batched
        full_batched(curvature, d)
    else
        full_unbatched(curvature, d)
    end
end

"""
    full(curvature::EmpiricalFisher, d::Union{Tuple,NamedTuple})

Compute the full empirical Fisher for batch of inputs-outputs, with the batch dimension at the end.
"""
function full_unbatched(curvature::EmpiricalFisher, d::Tuple)
    x, y = d

    loss = curvature.factor * curvature.loss_fun(x, y)
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

function full_batched(curvature::EmpiricalFisher, d::Tuple)
    x, y = d

    loss = curvature.factor * curvature.loss_fun(x, y)
    grads::Zygote.Grads = jacobian(
        () -> curvature.loss_fun(x, y; agg=identity), Flux.params(curvature.model)
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
