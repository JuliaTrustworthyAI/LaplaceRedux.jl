using Flux
using ..LaplaceRedux: get_loss_fun, outdim
using LinearAlgebra
using Zygote
using Tullio
using Compat

import Base: +, *, ==, length, getindex
import LinearAlgebra: det, logdet

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
    nn = curvature.model
    𝐠 = gradient(() -> curvature.loss_fun(nn(X), y), Flux.params(nn))           # compute the gradients of the loss function with respect to the model parameters
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

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)

    𝐉, fμ = jacobians(curvature, x)

    if curvature.likelihood == :regression
        H = 𝐉' * 𝐉
    else
        p = outdim(nn) > 1 ? softmax(fμ) : sigmoid(fμ)
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

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)

    𝐉, fμ = jacobians_batched(curvature, x)

    if curvature.likelihood == :regression
        @tullio H[i, j] := 𝐉[k, i, b] * 𝐉[k, j, b]
    else
        p = outdim(nn) > 1 ? softmax(fμ) : sigmoid(fμ)
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

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)
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

    nn = curvature.model
    loss = curvature.factor * curvature.loss_fun(nn(x), y)
    nn = curvature.model
    grads::Zygote.Grads = jacobian(
        () -> curvature.loss_fun(nn(x), y; agg=identity), Flux.params(nn)
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

"""
Macro for zero-based indexing. Example usage: (@zb A[0]) = ...
"""
macro zb(expr)
    if expr.head == :ref
        a, i = expr.args
        return Expr(:ref, esc(a), Expr(:call, :+, esc(i), 1))
    else
        error("Expected expr of form a[i], got: ($(expr)) with head $(expr.head)")
    end
end

struct Kron
    kfacs::Vector{Tuple{AbstractArray,AbstractArray}}
end

function (+)(l::Kron, r::Kron)
    @assert length(l.kfacs) == length(r.kfacs)
    kfacs = [
        Tuple(Hi + Hj for (Hi, Hj) in zip(Fi, Fj)) for (Fi, Fj) in zip(l.kfacs, r.kfacs)
    ]
    return Kron(kfacs)
end

function (==)(l::Kron, r::Kron)
    return l.kfacs == r.kfacs
end

function (*)(l::Real, r::Kron)
    kfacs = [Tuple(^(l, 1 / length(F)) * Hi for Hi in F) for F in r.kfacs]
    return Kron(kfacs)
end

(*)(l::Kron, r::Real) = (*)(r, l)

function getindex(K::Kron, i::Int)
    return K.kfacs[i]
end

"""
Interleave elements of multiple iterables in order provided.
"""
function interleave(iters...)
    return (elem for pack in zip(iters...) for elem in pack)
end

"""
Compute KFAC for the Fisher.
"""
function kron(
    curvature::Union{GGN,EmpiricalFisher}, subset_of_weights, data; batched::Bool=false
)
    @assert !isempty(data)
    @assert subset_of_weights != :subnetwork "Subnetwork Laplace requires a full or diagonal Hessian approximation!"
    # `d` is a zero-indexed array with layers sizes
    # `_zb` marks zero-based arrays: these should be accessed via the @zb macro
    xs = (pair[1] for pair in data)
    # We use the first element to initialise the shapes
    x_1 = first(xs)

    nn = curvature.model
    # NOTE: this method is for classification only, thus the loss must be logitcrossentropy
    # lossf = Flux.Losses.logitcrossentropy

    d_zb = [[size(x_1)]; map(a -> size(a), collect(Flux.activations(nn, x_1)))]
    n_layers = length(nn.layers)
    if subset_of_weights == :last_layer
        initial_layer = n_layers # designate the last layer as the initial layer
    else
        initial_layer = 1
    end

    double(sz) = (sz[1], sz[1])

    G_exp = [zeros(double(@zb d_zb[i])) for i in initial_layer:n_layers]
    # A separate matrix for bias-based gradients.
    G_exp_b = [zeros(double(@zb d_zb[i])) for i in initial_layer:n_layers]
    A_exp_zb = [zeros(double(@zb d_zb[i])) for i in (initial_layer - 1):(n_layers - 1)]

    # The data iterator is modelled lazily, so the number of samples is counted.
    n_data = 0

    for x_n in xs
        n_data += 1

        a_zb = [[x_n]; collect(Flux.activations(nn, x_n))]
        p = softmax(nn(x_n))

        # Approximate the expected value of the activation outer product A = aa'
        # across all samples x_n,
        # from the input to the pen-ultimate layer activation.
        A_exp_zb += [
            (@zb a_zb[i]) * transpose(@zb a_zb[i]) for
            i in (initial_layer - 1):(n_layers - 1)
        ]

        # Approx. the exp. value of the gradient (wrt layer non-activated output) outer product G = gg'
        # via the model's predictive distribution.
        for (j, yhat) in enumerate(eachcol(I(length(p))))
            loss_m = m -> curvature.loss_fun(m(x_n), yhat)
            grad, = gradient(loss_m, nn)

            # See Martens & Grosse 2015 page 5
            # DW[i] <- g[i] * a[i-1]'
            # In our case grads is DW
            g = [
                grad.layers[i].weight * pinv(transpose(@zb a_zb[i - 1])) for
                i in initial_layer:n_layers
            ]

            G = p[j] .* [g[i] * transpose(g[i]) for i in 1:length(g)]
            G_exp += G
            G_exp_b += G
        end
    end

    # Downscale the sums for A and G by the number of samples.
    # Only one of the factors is downscaled by N (like in laplace-torch)
    A_exp_zb /= n_data

    # The activation for the bias is simply one.
    # TODO: make Kron.kfacs a union type and include only the G
    A_exp_b_zb = [[1;;] for _ in initial_layer:n_layers]
    # Q: why are the factors not scaled in pytorch? bug?
    # G_exp_b /= N

    loss_xy = (x, y) -> curvature.loss_fun(nn(x), y)
    loss = sum(d -> loss_xy(d...), data)
    decomposed = decompose(
        Kron(collect(interleave(zip(G_exp, A_exp_zb), zip(G_exp_b, A_exp_b_zb))))
    )

    # NOTE: order is G, A, as in laplace-torch
    return loss, decomposed, n_data
end

struct KronDecomposed
    # TODO union types
    # kfacs :: Union{Vector{Tuple{AbstractArray, AbstractArray}},Vector{Matrix},Nothing}
    # kfacs :: Vector{Tuple{AbstractArray, AbstractArray}}
    kfacs::Vector{Tuple{Eigen,Eigen}}
    delta::Number
end

function clamp(eig::Eigen)
    return Eigen(max.(0, eig.values), eig.vectors)
end

"""
    decompose(K::Kron)

Eigendecompose Kronecker factors and turn into `KronDecomposed`.
"""
function decompose(K::Kron)
    # TODO filter out negative eigenvals
    return KronDecomposed(map(b -> map(clamp ∘ eigen, b), K.kfacs), 0)
end

"""
Shift the factors by a scalar across the diagonal.
"""
function (+)(K::KronDecomposed, delta::Number)
    return KronDecomposed(K.kfacs, K.delta + delta)
end

"""
Shift the factors by a diagonal (assumed uniform scaling)
"""
function (+)(K::KronDecomposed, delta::Diagonal)
    return K + first(delta)
end

"""
Multiply by a scalar by changing the eigenvalues. Distribute the scalar along the factors of a block.
"""
function (*)(K::KronDecomposed, scalar::Number)
    return KronDecomposed(
        map(
            b::Tuple ->
                map(e::Eigen -> Eigen(e.values * ^(scalar, 1 / length(b)), e.vectors), b),
            K.kfacs,
        ),
        K.delta,
    )
end

function (length)(K::KronDecomposed)
    return length(K.kfacs)
end

function getindex(K::KronDecomposed, i::Int)
    return K.kfacs[i]
end

# Commutative operations
(+)(delta::Number, K::KronDecomposed) = (+)(K::KronDecomposed, delta::Number)
(*)(scalar::Number, K::KronDecomposed) = (*)(K::KronDecomposed, scalar::Number)

"""
    logdetblock(block::Tuple{Eigen,Eigen}, delta::Number)

Log-determinant of a block in KronDecomposed, shifted by delta by on the diagonal.
"""
function logdetblock(block::Tuple{Eigen,Eigen}, delta::Number)
    L1, L2 = block[1].values, block[2].values
    return sum(log(L1 * transpose(L2) .+ delta))
end

"""
    logdet(K::KronDecomposed)

Log-determinant of the KronDecomposed block-diagonal matrix, as the product of the determinants of the blocks
"""
function logdet(K::KronDecomposed)
    return sum(b -> logdetblock(b, K.delta), K.kfacs)
end

"""
    det(K::KronDecomposed)

Log-determinant of the KronDecomposed block-diagonal matrix, as the exponentiated log-determinant.
"""
function det(K::KronDecomposed)
    return exp(logdet(K))
end

"""
Matrix-multuply for the KronDecomposed Hessian approximation K and a 2-d matrix W,
applying an exponent to K and transposing W before multiplication.
Return `(K^x)W^T`, where `x` is the exponent.
"""
function mm(K::KronDecomposed, W; exponent::Number=-1)
    cur_idx = 1

    @assert length(size(W)) == 2
    # If W is the Jacobian matrix of classifier NN output, then 
    # k - the number of classes 
    # p - number of params
    k, p = size(W)
    M = []
    for block in K.kfacs
        # Iterate block by block, compute the multiplication of this block
        # with the corresponding slice of W, -- the products are independent by block.
        Q1, Q2 = block[1].vectors, block[2].vectors
        L1, L2 = block[1].values, block[2].values
        # NOTE: Order of factors is reversed, like in laplace-torch, (G,A instead of A,G)
        # NOTE: Should this not be the other way?
        p_in = length(L1)
        p_out = length(L2)
        sz = p_in * p_out

        # TODO: find a reference to the derivations
        ldelta_exp = (L1 * transpose(L2) .+ K.delta) .^ exponent

        W_p = reshape(W[:, cur_idx:(cur_idx + sz - 1)], k, p_in, p_out)
        # This code closely mimics the laplace-torch Python counter-part (KronDecomposed._bmm),
        # however, there may be a better way to translate the operation.
        # Julia matrices are col-major, Numpy & Pytorch matrices are row-major,
        # hence the awkward permutedims.
        W_p = permutedims(
            stack([
                Q1 * ((transpose(Q1) * W_p[i, :, :] * Q2) .* ldelta_exp) * transpose(Q2) for
                i in range(1, size(W_p, 1))
            ]),
            (3, 1, 2),
        )
        W_p = reshape(W_p, k, sz)
        push!(M, W_p)

        # Slide the pointer forward by the size of the block
        cur_idx += sz
    end
    # The by-block products are concatenated to the full product.
    M = reduce(hcat, M)
    return M
end
