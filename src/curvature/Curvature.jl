module Curvature

using Compat
using Flux
using ..LaplaceRedux: get_loss_fun, outdim
using LinearAlgebra
using Tullio
using Zygote

export CurvatureInterface

"Base type for any curvature interface."
abstract type CurvatureInterface end

function Base.:(==)(a::CurvatureInterface, b::CurvatureInterface)
    checks = [getfield(a, x) == getfield(b, x) for x in fieldnames(typeof(a))]
    println(checks)
    return all(checks)
end

include("utils.jl")
include("ggn.jl")
include("fisher.jl")

end
