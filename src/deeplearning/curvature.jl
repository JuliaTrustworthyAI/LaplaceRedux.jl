using Flux

abstract type CurvatureInterface end
struct EFInterface <: CurvatureInterface
    model::Any
    likelihood::Symbol
end

function full(∇∇::EFInterface; d)
    𝐠 = gradient(params(∇∇.model)) do 
        l = loss(d...)
    end
    𝐇 = 𝐠 * 𝐠'
    return 𝐇
end