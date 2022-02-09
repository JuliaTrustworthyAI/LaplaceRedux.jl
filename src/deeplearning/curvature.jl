abstract type CurvatureInterface end

struct EFInterface <: CurvatureInterface
    model::Any
    likelihood::Symbol
end

function full(∇∇::EFInterface; x, y)
    𝐠 = gradients(∇∇, x, y)
    𝐇 = 𝐠 * 𝐠'
    return 𝐇
end