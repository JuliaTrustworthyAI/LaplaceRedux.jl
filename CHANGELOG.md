# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v0.2.1].
## Version [0.2.2] - 2024-05-30

### Changed

- Unified duplicated function MMI.clean!: previously MMI.clean! consisted of two separate functions for handling :classification and :regression types respectively. Now, a single MMI.clean! function handles both cases efficiently.[#39]
- Split LaplaceApproximation struct in two different structs:LaplaceClassification and LaplaceRegression  [#39] 
- Unified the MLJFlux.shape and the MLJFlux.build functions to handle both :classification and :regression tasks. In particular, shape now handles multi-output regression cases too [[#39](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/issues/39)]

### Added
 



## Version [0.2.1] - 2024-05-29

### Changed

- Improved the docstring for the `predict` and `glm_predictive_distribution` methods. [#88]

### Added

- Added `probit` helper function to compute probit approximation for classification. [#88]