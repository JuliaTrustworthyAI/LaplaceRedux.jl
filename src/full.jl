"""
    approximate(curvature::CurvatureInterface, hessian_structure::FullHessian, d::Tuple; batched::Bool=false)

Compute the full approximation, for either a single input-output datapoint or a batch of such. 
"""
function approximate(
    curvature::CurvatureInterface,
    hessian_structure::FullHessian,
    d::Tuple;
    batched::Bool=false,
)
    if batched
        Curvature.full_batched(curvature, d)
    else
        Curvature.full_unbatched(curvature, d)
    end
end

"""
    _fit!(la::Laplace, hessian_structure::FullHessian, data; batched::Bool=false, batchsize::Int, override::Bool=true)

Fit a Laplace approximation to the posterior distribution of a model using the full Hessian.
"""
function _fit!(
    la::Laplace,
    hessian_structure::FullHessian,
    data;
    batched::Bool=false,
    batchsize::Int,
    override::Bool=true,
)
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
    la.posterior.H = H
    la.posterior.loss = loss
    la.posterior.P = posterior_precision(la)
    la.posterior.Σ = posterior_covariance(la, la.posterior.P)
    return la.posterior.n_data = n_data
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
