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

include("utils.jl")
include("ggn.jl")
include("fisher.jl")

end
