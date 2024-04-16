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
"""
function glm_predictive_distribution(la::AbstractLaplace, X::AbstractArray)
    𝐉, fμ = Curvature.jacobians(la.est_params.curvature, X)
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

"""
    predict(la::AbstractLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true)

Computes predictions from Bayesian neural network.

# Examples

```julia-repl
using Flux, LaplaceRedux
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn)
fit!(la, data)
predict(la, hcat(x...))
```
"""
function predict(
    la::AbstractLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true
)
    fμ, fvar = glm_predictive_distribution(la, X)

    # Regression:
    if la.likelihood == :regression
        return fμ, fvar
    end

    # Classification:
    if la.likelihood == :classification

        # Probit approximation
        if link_approx == :probit
            κ = 1 ./ sqrt.(1 .+ π / 8 .* fvar)
            z = κ .* fμ
        end

        if link_approx == :plugin
            z = fμ
        end

        # Sigmoid/Softmax
        if predict_proba
            if la.posterior.n_out == 1
                p = Flux.sigmoid(z)
            else
                p = Flux.softmax(z; dims=1)
            end
        else
            p = z
        end

        return p
    end
end

"""
    predict(la::AbstractLaplace, X::Matrix; link_approx=:probit, predict_proba::Bool=true)

Compute predictive posteriors for a batch of inputs.

Predicts on a matrix of inputs. Note, input is assumed to be batched only if it is a matrix.
If the input dimensionality of the model is 1 (a vector), one should still prepare a 1×B matrix batch as input.
"""
function predict(
    la::AbstractLaplace, X::Matrix; link_approx=:probit, predict_proba::Bool=true
)
    return stack([
        predict(la, X[:, i]; link_approx=link_approx, predict_proba=predict_proba) for
        i in 1:size(X, 2)
    ])
end

"""
    (la::AbstractLaplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::AbstractLaplace)(X::AbstractArray; kwrgs...)
    return predict(la, X; kwrgs...)
end
