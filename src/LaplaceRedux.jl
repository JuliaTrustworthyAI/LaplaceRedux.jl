module LaplaceRedux

include("utils.jl")

include("data/Data.jl")
using .Data

include("curvature/Curvature.jl")
using .Curvature

include("baselaplace.jl")       # abstract base type and methods
include("laplace.jl")           # full Laplace

export Laplace, fit!, predict, optimize_prior!, glm_predictive_distribution

include("plotting.jl")

end
