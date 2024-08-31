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

"""
    functional_variance(la::AbstractLaplace, 𝐉::AbstractArray)

Compute the functional variance for the GLM predictive. Dispatches to the appropriate method based on the Hessian structure.
"""
function functional_variance(la, 𝐉)
    return functional_variance(la, la.est_params.hessian_structure, 𝐉)
end

"""
    glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)

Computes the linearized GLM predictive.

# Arguments

- `la::AbstractLaplace`: A Laplace object.
- `X::AbstractArray`: Input data.

# Returns
- `normal_distr` A normal distribution N(fμ,fvar) approximating the predictive distribution p(y|X) given the input data X.
- `fμ::AbstractArray`: Mean of the predictive distribution. The output shape is column-major as in Flux.
- `fvar::AbstractArray`: Variance of the predictive distribution. The output shape is column-major as in Flux.

# Examples

```julia-repl
using Flux, LaplaceRedux
using LaplaceRedux.Data: toy_data_linear
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn; likelihood=:classification)
fit!(la, data)
glm_predictive_distribution(la, hcat(x...))
"""
function glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.est_params.curvature, X)
    fμ = reshape(fμ, Flux.outputsize(la.model, size(X)))
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    pred_fvar = fvar .^ 2 .+ la.prior.σ^2
    fstd = sqrt.(pred_fvar)
    normal_distr = [Normal(fμ[i], fstd[i]) for i in 1:size(fμ, 2)]
    return (normal_distr, fμ, pred_fvar)
end

"""
    predict(la::AbstractLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true)

Computes predictions from Bayesian neural network.

# Arguments

- `la::AbstractLaplace`: A Laplace object.
- `X::AbstractArray`: Input data.
- `link_approx::Symbol=:probit`: Link function approximation. Options are `:probit` and `:plugin`.
- `predict_proba::Bool=true`: If `true` (default) apply a sigmoid or a softmax function to the output of the Flux model.
- `return_distr::Bool=false`: if `false` (default), the function output either the direct output of the chain or pseudo-probabilities (if predict_proba= true).
    if `true` predict return a Bernoulli distribution in binary classification tasks and a categorical distribution in multiclassification tasks.

# Returns
For classification tasks, LaplaceRedux provides different options:
if ret_distr is false:
    - `fμ::AbstractArray`: Mean of the predictive distribution if link function is set to `:plugin`, otherwise the probit approximation. The output shape is column-major as in Flux.
if ret_distr is true:
    - a Bernoulli distribution in binary classification tasks and a categorical distribution in multiclassification tasks.
For regression tasks:
- `normal_distr::Distributions.Normal`:the array of Normal distributions computed by glm_predictive_distribution. 
# Examples

```julia-repl
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
    normal_distr, fμ, fvar = glm_predictive_distribution(la, X)

    # Regression:
    if la.likelihood == :regression
        if ret_distr
            return reshape(normal_distr, (:, 1))
        else
            return fμ, fvar
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
