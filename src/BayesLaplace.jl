module BayesLaplace

include("curvature/Curvature.jl")
using .Curvature
include("laplace.jl")

export laplace, fit!, predict, plugin

end
