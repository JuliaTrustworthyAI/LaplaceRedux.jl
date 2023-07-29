mutable struct KronLaplace <: BaseLaplace
    @fields_baselaplace
end

function _fit!(
    la::KronLaplace, data; batched::Bool=false, batchsize::Int, override::Bool=true
)
    @assert !batched "Batched Kronecker-factored Laplace approximations not supported"
    @assert la.likelihood == :classification &&
        get_loss_type(la.likelihood, la.curvature.model) == :logitcrossentropy "Only multi-class classification supported"

    # NOTE: the fitting process is structured differently for Kronecker-factored methods
    # to avoid allocation, initialisation & interleaving overhead
    # Thus the loss, Hessian, and data size is computed not in a loop but in a separate function.
    loss, H, n_data = Curvature.kron(la.curvature, data; batched=batched)

    la.loss = loss
    la.H = H
    la.P = posterior_precision(la)
    # NOTE: like in laplace-torch, post covariance is not defined for KronLaplace
    return la.n_data = n_data
end

"""
functional_variance(la::KronLaplace, 𝐉::Matrix)

Compute functional variance for the GLM predictive: as the diagonal of the K×K predictive output covariance matrix 𝐉𝐏⁻¹𝐉ᵀ,
where K is the number of outputs, 𝐏 is the posterior precision, and 𝐉 is the Jacobian of model output `𝐉=∇f(x;θ)|θ̂`.
"""
function functional_variance(la::KronLaplace, 𝐉::Matrix)
    return diag(inv_square_form(la.P, 𝐉))
end

"""
function inv_square_form(K::KronDecomposed, W::Matrix)

Special function to compute the inverse square form 𝐉𝐏⁻¹𝐉ᵀ (or 𝐖𝐊⁻¹𝐖ᵀ)
"""
function inv_square_form(K::KronDecomposed, W::Matrix)
    SW = mm(K, W; exponent=-1)
    return W * SW'
end
