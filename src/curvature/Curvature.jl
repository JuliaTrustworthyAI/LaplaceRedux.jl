module Curvature

using Flux
using ..LaplaceRedux: get_loss_fun, outdim
using LinearAlgebra
using Zygote
using Tullio
using Compat

import Base: +, *, ==, length, getindex
import LinearAlgebra: det, logdet

"Basetype for any curvature interface."
abstract type CurvatureInterface end

include("utils.jl")
include("ggn.jl")
include("fisher.jl")
include("approximations/approximations.jl")

end
