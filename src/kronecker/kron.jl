include("utils.jl")

"Concrete type for Kronecker-factored Hessian structure."
struct KronHessian <: HessianStructure end

"""
    approximate(curvature::CurvatureInterface, hessian_structure::KronHessian, data; batched::Bool=false)

Compute the eigendecomposed Kronecker-factored approximate curvature as the Fisher information matrix.

Note, since the network predictive distribution is used in a weighted sum, and the number of backward
passes is linear in the number of target classes, e.g. 100 for CIFAR-100.
"""
function approximate(
    curvature::CurvatureInterface, hessian_structure::KronHessian, data; batched::Bool=false
)
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
    _fit!(la::Laplace, hessian_structure::KronHessian, data; batched::Bool=false, batchsize::Int, override::Bool=true)

Fit a Laplace approximation to the posterior distribution of a model using the Kronecker-factored Hessian.
"""
function _fit!(
    la::Laplace,
    hessian_structure::KronHessian,
    data;
    batched::Bool=false,
    batchsize::Int,
    override::Bool=true,
)
    @assert !batched "Batched Kronecker-factored Laplace approximations not supported"
    @assert la.likelihood == :classification &&
        get_loss_type(la.likelihood, la.est_params.curvature.model) ==
            :logitcrossentropy "Only multi-class classification supported"

    # NOTE: the fitting process is structured differently for Kronecker-factored methods
    # to avoid allocation, initialisation & interleaving overhead
    # Thus the loss, Hessian, and data size is computed not in a loop but in a separate function.
    loss, H, n_data = approximate(
        la.est_params.curvature, hessian_structure, data; batched=batched
    )

    # Store output:
    la.posterior.H = H
    la.posterior.loss = loss
    la.posterior.P = posterior_precision(la)
    return la.posterior.n_data = n_data
    # NOTE: like in laplace-torch, post covariance is not defined for KronLaplace
end

"""
    functional_variance(la::Laplace, hessian_structure::KronHessian, 𝐉::Matrix)

Compute functional variance for the GLM predictive: as the diagonal of the K×K predictive output covariance matrix 𝐉𝐏⁻¹𝐉ᵀ,
where K is the number of outputs, 𝐏 is the posterior precision, and 𝐉 is the Jacobian of model output `𝐉=∇f(x;θ)|θ̂`.
"""
function functional_variance(la::Laplace, hessian_structure::KronHessian, 𝐉::Matrix)
    return diag(inv_square_form(la.posterior.P, 𝐉))
end

"""
function inv_square_form(K::KronDecomposed, W::Matrix)

Special function to compute the inverse square form 𝐉𝐏⁻¹𝐉ᵀ (or 𝐖𝐊⁻¹𝐖ᵀ)
"""
function inv_square_form(K::KronDecomposed, W::Matrix)
    SW = mm(K, W; exponent=-1)
    return W * SW'
end
