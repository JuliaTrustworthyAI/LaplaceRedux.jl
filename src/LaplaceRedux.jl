module LaplaceRedux

include("data/Data.jl")
using .Data

include("curvature/Curvature.jl")
using .Curvature

include("laplace.jl")

export Laplace, fit!, predict, plugin

end
