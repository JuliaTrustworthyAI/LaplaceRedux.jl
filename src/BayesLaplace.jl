module BayesLaplace

include("logit.jl")
include("Curvature.jl")
using .Curvature
include("laplace.jl")

export bayes_logreg, laplace, fit!, predict

end
