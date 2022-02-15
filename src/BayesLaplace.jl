module BayesLaplace

include("logit.jl")
include("Curvature.jl")
using .Curvature
include("laplace.jl")
include("utils.jl")

export bayes_logreg, laplace, fit!, predict,
        plot_data!, plot_contour, toy_data_linear, toy_data_non_linear

end
