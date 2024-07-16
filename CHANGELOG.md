# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v0.2.1].


## Version [1.0.0] - 2024-07-16

### Changed

-Changed the behavior of the `predict` function so that it now gives the user the possibility to get distributions from the Distributions.jl package as output. [#99]

### Added

- Added new keyword parameter ret_distr::Bool=false to predict. [#99]

## Version [0.2.1] - 2024-05-29

### Changed

- Improved the docstring for the `predict` and `glm_predictive_distribution` methods. [#88]

### Added

- Added `probit` helper function to compute probit approximation for classification. [#88]