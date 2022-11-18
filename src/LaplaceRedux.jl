module LaplaceRedux

include("utils.jl")

include("data/Data.jl")
using .Data

include("curvature/Curvature.jl")
using .Curvature

include("laplace.jl")

export Laplace, fit!, predict, plugin

include("plotting.jl")

end
