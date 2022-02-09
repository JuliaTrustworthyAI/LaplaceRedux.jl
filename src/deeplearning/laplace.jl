using Flux

function laplace(model::Any; likelihood=:classification, subset_of_weights=:all, hessian_structure=:full, backend=EFInterface) 
    return FullLaplace(model, likelihood, backend, zeros())
end

abstract type BaseLaplace end

function fit(𝑳::BaseLaplace,data)
    for d in data
        𝐇 = _curv_closure(𝑳, d)
    end
    return 𝐇
end

struct FullLaplace <: BaseLaplace
    model::Any
    likelihood::Symbol
    backend::CurvatureInterface
    𝐇::AbstractArray
end

function _curv_closure(𝑳::FullLaplace, d)
    𝐇 = full(𝑳.backend, d)
    return 𝐇
end


