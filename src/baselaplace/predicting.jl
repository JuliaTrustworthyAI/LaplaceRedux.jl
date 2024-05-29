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

- `fμ::AbstractArray`: Mean of the predictive distribution. The format is column-major as in Flux.
- `fvar::AbstractArray`: Variance of the predictive distribution. The format is column-major as in Flux.

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
    fμ = permutedims(fμ)
    fvar = functional_variance(la, 𝐉)
    fvar = reshape(fvar, size(fμ)...)
    return fμ, fvar
end

"""
    predict(la::AbstractLaplace, X::AbstractArray; link_approx=:probit, predict_proba::Bool=true)

Computes predictions from Bayesian neural network.

# Arguments

- `la::AbstractLaplace`: A Laplace object.
- `X::AbstractArray`: Input data.
- `link_approx::Symbol=:probit`: Link function approximation. Options are `:probit` and `:plugin`.
- `predict_proba::Bool=true`: If `true` (default), returns probabilities for classification tasks.

# Returns

- `fμ::AbstractArray`: Mean of the predictive distribution if link function is set to `:plugin`, otherwise the probit approximation. The format is column-major as in Flux.

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
            z = probit(fμ, fvar)
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
    probit(fμ::AbstractArray, fvar::AbstractArray)

Compute the probit approximation of the predictive distribution.
"""
function probit(fμ::AbstractArray, fvar::AbstractArray)
    κ = 1 ./ sqrt.(1 .+ π / 8 .* fvar)
    z = κ .* fμ
    return z
end

"""
    (la::AbstractLaplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::AbstractLaplace)(X::AbstractArray; kwrgs...)
    return predict(la, X; kwrgs...)
end
