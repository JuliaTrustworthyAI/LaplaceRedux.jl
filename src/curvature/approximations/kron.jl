"""
Macro for zero-based indexing. Example of usage: (@zb A[0]) = ...
"""
macro zb(expr)
    if expr.head == :ref
        a, i = expr.args
        return Expr(:ref, esc(a), Expr(:call, :+, esc(i), 1))
    else
        error("Expected expr of form a[i], got: ($(expr)) with head $(expr.head)")
    end
end

"""
Kronecker-factored approximate curvature representation for a neural network model.
Each element in kfacs represents two Kronecker factors (𝐆, 𝐀), such that the full block Hessian approximation would be approximated as 𝐀⊗𝐆.
"""
struct Kron
    kfacs::Vector{Tuple{AbstractArray,AbstractArray}}
end

"""
Kronecker-factored curvature sum.
"""
function (+)(l::Kron, r::Kron)
    @assert length(l.kfacs) == length(r.kfacs)
    kfacs = [
        Tuple(Hi + Hj for (Hi, Hj) in zip(Fi, Fj)) for (Fi, Fj) in zip(l.kfacs, r.kfacs)
    ]
    return Kron(kfacs)
end

"""
Kronecker-factored curvature equality.
"""
function (==)(l::Kron, r::Kron)
    return l.kfacs == r.kfacs
end

"""
Kronecker-factored curvature scalar scaling.
"""
function (*)(l::Real, r::Kron)
    kfacs = [Tuple(^(l, 1 / length(F)) * Hi for Hi in F) for F in r.kfacs]
    return Kron(kfacs)
end

# Commutative operation
(*)(l::Kron, r::Real) = (*)(r, l)

"""
Get Kronecker-factored block represenation.
"""
function getindex(K::Kron, i::Int)::Tuple{AbstractArray,AbstractArray}
    return K.kfacs[i]
end

"""
Interleave elements of multiple iterables in order provided.
"""
function interleave(iters...)
    return (elem for pack in zip(iters...) for elem in pack)
end

"""
Compute the eigendecomposed Kronecker-factored approximate curvature as the Fisher information matrix.

Note, since the network predictive distribution is used in a weighted sum, and the number of backward
passes is linear in the number of target classes, e.g. 100 for CIFAR-100.
"""
function kron(curvature::CurvatureInterface, data; batched::Bool=false)
    subset_of_weights = curvature.subset_of_weights
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

    # The data iterator is modeled lazily, so the number of samples is counted.
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

"""
Decomposed Kronecker-factored approximate curvature representation for a neural network model.

Decomposition is required to add the prior (diagonal matrix) to the posterior (`KronDecomposed`).
It also has the benefits of reducing the costs for computation of inverses and log-determinants.
"""
struct KronDecomposed
    # TODO union types
    # kfacs :: Union{Vector{Tuple{AbstractArray, AbstractArray}},Vector{Matrix},Nothing}
    # kfacs :: Vector{Tuple{AbstractArray, AbstractArray}}
    kfacs::Vector{Tuple{Eigen,Eigen}}
    delta::Number
end

"""
Clamp eigenvalues in an eigendecomposition to be non-negative.

Since the Fisher information matrix is a positive-semidefinite by construction, the (near-zero) negative
eigenvalues should be neglected.
"""
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

"""
Number of blocks in a Kronecker-factored curvature.
"""
function (length)(K::KronDecomposed)
    return length(K.kfacs)
end

"""
Get i-th block of a a Kronecker-factored curvature.
"""
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
