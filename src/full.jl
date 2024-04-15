"""
    _fit!(la::Laplace, hessian_structure::FullHessian, data; batched::Bool=false, batchsize::Int, override::Bool=true)

Fit a Laplace approximation to the posterior distribution of a model using the full Hessian.
"""
function _fit!(la::Laplace, hessian_structure::FullHessian, data; batched::Bool=false, batchsize::Int, override::Bool=true)
    if override
        H = _init_H(la)
        loss = 0.0
        n_data = 0
    end

    for d in data
        loss_batch, H_batch = hessian_approximation(la, d; batched=batched)
        loss += loss_batch
        H += H_batch
        n_data += batchsize
    end

    # Store output:
    la.loss = loss
    # Hessian
    la.H = H
    # Posterior precision
    la.P = posterior_precision(la)
    # Posterior covariance
    la.Σ = posterior_covariance(la, la.P)
    la.curvature.params = get_params(la)
    # Number of observations
    return la.n_data = n_data
end

"""
functional_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.
"""
function functional_variance(la::Laplace, hessian_structure::FullHessian, 𝐉)
    Σ = posterior_covariance(la)
    fvar = map(j -> (j' * Σ * j), eachrow(𝐉))
    return fvar
end

