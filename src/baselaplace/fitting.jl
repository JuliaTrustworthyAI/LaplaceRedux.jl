"""
    hessian_approximation(la::AbstractLaplace, d; batched::Bool=false)

Computes the local Hessian approximation at a single datapoint `d`.
"""
function hessian_approximation(la::AbstractLaplace, d; batched::Bool=false)
    loss, H = approximate(
        la.est_params.curvature, la.est_params.hessian_structure, d; batched=batched
    )
    return loss, H
end

"""
    fit!(la::AbstractLaplace,data)

Fits the Laplace approximation for a data set.
The function returns the number of observations (n_data) that were used to update the Laplace object.
It does not return the updated Laplace object itself because the function modifies the input Laplace object in place (as denoted by the use of '!' in the function's name).

# Examples

```julia-repl
using Flux, LaplaceRedux
x, y = LaplaceRedux.Data.toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn)
fit!(la, data)
```

"""
function fit!(la::AbstractLaplace, data; override::Bool=true)
    return _fit!(
        la,
        la.est_params.hessian_structure,
        data;
        batched=false,
        batchsize=1,
        override=override,
    )
end

"""
Fit the Laplace approximation, with batched data.
"""
function fit!(la::AbstractLaplace, data::DataLoader; override::Bool=true)
    return _fit!(
        la,
        la.est_params.hessian_structure,
        data;
        batched=true,
        batchsize=data.batchsize,
        override=override,
    )
end
