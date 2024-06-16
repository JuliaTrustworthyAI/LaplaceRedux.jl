# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v0.2.1].

## Version [0.3.0] - 2024-06-8

### Changed

- Updated codecov workflow in CI.yml. [#39]
- fixed test functions [#39]
- adapted the LaplaceClassification and the LaplaceRegression struct to use the new @mlj_model macro from MLJBase.[#39]
- Changed the fit! method arguments. [#39]
- Changed the predict functions for both LaplaceClassification and  LaplaceRegression.[#39]

### Removed

- Removed the shape, build and clean! functions.[#39]
- Removed Review dog for code format suggestions. [#39]

## Version [0.2.3] - 2024-05-31

### Changed

- Removed the link_approx parameter in LaplaceRegression since it is not required.
- Changed MMI.clean! to check the value of link_approx only in the case likelihood is set to `:classification`
- Now the likelihood type in LaplaceClassification and LaplaceRegression is automatically set by the inner constructor. The user is not required to provide it as a parameter anymore.




## Version [0.2.2] - 2024-05-30

### Changed

- Unified duplicated function MMI.clean!: previously MMI.clean! consisted of two separate functions for handling :classification and :regression types respectively. Now, a single MMI.clean! function handles both cases efficiently.[#39]
- Split LaplaceApproximation struct in two different structs:LaplaceClassification and LaplaceRegression  [#39] 
- Unified the MLJFlux.shape and the MLJFlux.build functions to handle both :classification and :regression tasks. In particular, shape now handles multi-output regression cases too [[#39](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/issues/39)]
- Changed model metadata for LaplaceClassification and LaplaceRegression

### Added
 Added Distributions to LaplaceRedux dependency ( needed for MMI.predict(model::LaplaceRegression, fitresult, Xnew) )



## Version [0.2.1] - 2024-05-29

### Changed

- Improved the docstring for the `predict` and `glm_predictive_distribution` methods. [#88]

### Added

- Added `probit` helper function to compute probit approximation for classification. [#88]