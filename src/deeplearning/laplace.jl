using Flux

function laplace(model::Any; likelihood=:classification, subset_of_weights=:all, hessian_structure=:full) 
    return FullLaplace(model; likelihood=likelihood)
end

abstract type BaseLaplace end

function fit(𝑳::BaseLaplace,data)
    𝐇 = _curv_closure(𝑳, data)
    return 𝐇
end

struct FullLaplace <: BaseLaplace
    model::Any
    likelihood::Symbol
end

function _curv_closure(𝑳::FullLaplace, data)
    full()
end


