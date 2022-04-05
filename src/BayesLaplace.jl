module BayesLaplace

include("Curvature.jl")
using .Curvature
include("laplace.jl")
include("utils.jl")

export laplace, fit!, predict,
        plot_data!, plot_contour, toy_data_linear, toy_data_non_linear

end
