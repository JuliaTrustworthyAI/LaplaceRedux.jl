using Distributions: Normal, Bernoulli, Categorical
using Flux
using Statistics: mean, var

"""
    has_softmax_or_sigmoid_final_layer(model::Flux.Chain)

Check if the FLux model ends with a sigmoid or with a softmax layer

Input:
    - `model`: the Flux Chain object that represent the neural network.
Return:
    - `has_finaliser`: true if the check is positive, false otherwise.

"""
function has_softmax_or_sigmoid_final_layer(model::Flux.Chain)
    # Get the last layer of the model
    last_layer = last(model.layers)

    # Check if the last layer is either softmax or sigmoid
    has_finaliser = (last_layer == Flux.sigmoid || last_layer == Flux.softmax)

    return has_finaliser
end

@doc raw"""
    functional_variance(la::AbstractLaplace, 𝐉::AbstractArray)

Computes the functional variance for the GLM predictive as `map(j -> (j' * Σ * j), eachrow(𝐉))` which is a (output x output) predictive covariance matrix. Formally, we have ``{\mathbf{J}_{\hat\theta}}^\intercal\Sigma\mathbf{J}_{\hat\theta}`` where ``\mathbf{J}_{\hat\theta}=\nabla_{\theta}f(x;\theta)|\hat\theta`` is the Jacobian evaluated at the MAP estimate.
    
Dispatches to the appropriate method based on the Hessian structure.
"""
function functional_variance(la, 𝐉)
    return functional_variance(la, la.est_params.hessian_structure, 𝐉)
end

@doc raw"""
    glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)

Computes the linearized GLM predictive from neural network with a Laplace approximation to the posterior ``p(\theta|\mathcal{D})=\mathcal{N}(\hat\theta,\Sigma)``. 
This is the distribution on network outputs given by ``p(f(x)|x,\mathcal{D})\approx \mathcal{N}(f(x;\hat\theta),{\mathbf{J}_{\hat\theta}}^\intercal\Sigma\mathbf{J}_{\hat\theta})``. 
For the Bayesian predictive distribution, see [`predict`](@ref).


# Arguments

- `la::AbstractLaplace`: A Laplace object.
- `X::AbstractArray`: Input data.

# Returns
- `normal_distr` A normal distribution N(fμ,fvar) approximating the predictive distribution p(y|X) given the input data X.- `normal_distr` A normal distribution N(fμ,fvar) approximating the predictive distribution p(y|X) given the input data X.
- `fμ::AbstractArray`: Mean of the predictive distribution. The output shape is column-major as in Flux.
- `fvar::AbstractArray`: Variance of the predictive distribution. The output shape is column-major as in Flux.

# Examples

```julia
using Flux, LaplaceRedux
using LaplaceRedux.Data: toy_data_linear
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn; likelihood=:classification)
fit!(la, data)
glm_predictive_distribution(la, hcat(x...))
```
"""
function glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.est_params.curvature, X)
    fμ = reshape(fμ, Flux.outputsize(la.model, size(X)))
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    fstd = sqrt.(fvar)
    normal_distr = [Normal(fμ[i], fstd[i]) for i in axes(fμ, 2)]
    return (normal_distr, fμ, fvar)
end

@doc raw"""
    predict(
        la::AbstractLaplace,
        X::AbstractArray;
        link_approx=:probit,
        predict_proba::Bool=true,
        ret_distr::Bool=false,
    )

Computes the Bayesian predictivie distribution from a neural network with a Laplace approximation to the posterior ``p(\theta | \mathcal{D}) = \mathcal{N}(\hat\theta, \Sigma)``.

# Arguments

- `la::AbstractLaplace`: A Laplace object.
- `X::AbstractArray`: Input data.
- `link_approx::Symbol=:probit`: Link function approximation. Options are `:probit` and `:plugin`.
- `predict_proba::Bool=true`: If `true` (default) apply a sigmoid or a softmax function to the output of the Flux model.
- `return_distr::Bool=false`: if `false` (default), the function outputs either the direct output of the chain or pseudo-probabilities (if `predict_proba=true`).
    if `true` predict returns a probability distribution.

# Returns

For classification tasks:

1. If `ret_distr` is `false`, `predict` returns `fμ`, i.e. the mean of the predictive distribution, which corresponds to the MAP estimate if the link function is set to `:plugin`, otherwise the probit approximation. The output shape is column-major as in Flux.
2. If `ret_distr` is `true`, `predict` returns a Bernoulli distribution in binary classification tasks and a categorical distribution in multiclassification tasks.

For regression tasks:

1. If `ret_distr` is `false`, `predict` returns the mean and the variance of the predictive distribution. The output shape is column-major as in Flux.
2. If `ret_distr` is `true`, `predict` returns the predictive posterior distribution, namely:

``p(y|x,\mathcal{D})\approx \mathcal{N}(f(x;\hat\theta),{\mathbf{J}_{\hat\theta}}^\intercal\Sigma\mathbf{J}_{\hat\theta} + \sigma^2 \mathbf{I})``

# Examples

```julia
using Flux, LaplaceRedux
using LaplaceRedux.Data: toy_data_linear
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn; likelihood=:classification)
fit!(la, data)
predict(la, hcat(x...))
```
"""
function predict(
    la::AbstractLaplace,
    X::AbstractArray;
    link_approx=:probit,
    predict_proba::Bool=true,
    ret_distr::Bool=false,
)
    _, fμ, fvar = glm_predictive_distribution(la, X)

    # Regression:
    if la.likelihood == :regression

        # Add observational noise:
        pred_var = fvar .+ la.prior.observational_noise^2
        fstd = sqrt.(pred_var)
        pred_dist = [Normal(fμ[i], fstd[i]) for i in axes(fμ, 2)]

        if ret_distr
            return reshape(pred_dist, (:, 1))
        else
            return fμ, pred_var
        end
    end

    # Classification:
    if la.likelihood == :classification
        has_finaliser = has_softmax_or_sigmoid_final_layer(la.model)

        # case when no softmax/sigmoid  function is applied
        if has_finaliser == false

            # Probit approximation
            if link_approx == :probit
                z = probit(fμ, fvar)
            end

            if link_approx == :plugin
                z = fμ
            end

            # Sigmoid/Softmax
            if predict_proba
                if la.posterior.n_out == 1
                    p = Flux.sigmoid(z)
                    if ret_distr
                        p = map(x -> Bernoulli(x), p)
                    end

                else
                    p = Flux.softmax(z; dims=1)
                    if ret_distr
                        p = mapslices(col -> Categorical(col), p; dims=1)
                    end
                end
            else
                if ret_distr
                    @warn "the model does not produce pseudo-probabilities. ret_distr will not work if predict_proba is set to false."
                end
                p = z
            end
        else # case when has_finaliser is true 
            if predict_proba == false
                @warn "the model already produce pseudo-probabilities since it has either sigmoid or a softmax layer as a final layer."
            end
            if ret_distr
                if la.posterior.n_out == 1
                    p = map(x -> Bernoulli(x), fμ)
                else
                    p = mapslices(col -> Categorical(col), fμ; dims=1)
                end

            else
                p = fμ
            end
        end

        return p
    end
end

"""
    probit(fμ::AbstractArray, fvar::AbstractArray)

Compute the probit approximation of the predictive distribution.
"""
function probit(fμ::AbstractArray, fvar::AbstractArray)
    κ = 1 ./ sqrt.(1 .+ π / 8 .* fvar)
    z = κ .* fμ
    return z
end

"""
    (la::AbstractLaplace)(X::AbstractArray)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::AbstractLaplace)(X::AbstractArray)
    return la.model(X)
end
