module LaplaceRedux

include("utils.jl")

include("data/Data.jl")
using .Data

include("curvature/Curvature.jl")
using .Curvature

include("baselaplace/core_struct.jl")
include("full.jl")
include("kronecker/kron.jl")

include("subnet.jl")

export Laplace
export fit!, predict
export optimize_prior!,
    glm_predictive_distribution, posterior_covariance, posterior_precision
export collect_trainable, get_params


include("calibration_functions.jl")
export empirical_frequency_binary_classification,
    sharpness_classification,
    empirical_frequency_regression,
    sharpness_regression,
    extract_mean_and_variance,
    sigma_scaling,
    rescale_stddev

include("direct_mlj.jl")
end
