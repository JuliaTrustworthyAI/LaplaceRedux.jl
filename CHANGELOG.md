# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v0.2.1].

## Version [1.3.0] - 2026-02-05

- Migrated to new Flux API (see #146).
- Updated compatibility bounds for various packages.
- Breaking change to internal API as a consequence of migration: `gradient` no longer returns `::Zygote.Grads`.

## Version [1.2.2] - 2026-01-08

- Merged a bunch of compat related PRs.

## Version [1.2.1] - 2025-12-19

- temporarily removed  TaijaData due to issues with CategoricalDistributions 0.2 [#142]
- Docs env now has compatibility issues with TajaPlotting and RData(needs to be fixed).    Cannot add CategoricalDistributions 0.2 without conflicts
- updated the package CategoricalDistributions to 0.2 in LaplaceRedux
- Explicitly used LaplaceRedux.Laplace in the pytorch_comparison.jl to avoid name conflicts


## Version [1.2.0] - 2024-12-03

### Changed

- Largely removed unicode characters from code base. [#134]
- Removed legacy v1.9 from CI testing. [#134]

### Added

- Added general support for MLJ [#126] [#134]

## Version [1.1.1] - 2024-09-12

### Changed

- Fixed an issue in MLJFlux implementation that led to long compute times for predictions. [#122]

## Version [1.1.0] - 2024-09-03

### Changed

- Predict function now returns predictive distribution that includes observational noise estimates for regression. [#116]

### Added

- Adds support for calibration. [#90]

## Version [1.0.2] - 2024-08-12

### Added 

- added TaijaPlotting to the docs env

### Changed

- modified the MLJFlux.train function so that it now properly return a trained chain [[#112](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/issues/112)]

## Version [1.0.0] - 2024-07-22

### Changed

- added the option to return meand and variance to predict in the case of regression[[#101](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/issues/101)]
- modified mlj_flux.jl by adding the ret_distr parameter and fixed mljflux.predict both for classification and regression tasks. 
- Changed the behavior of the `predict` function so that it now gives the user the possibility to get distributions from the Distributions.jl package as output. [#99]
- Calling a Laplace object on an array, `(la::AbstractLaplace)(X::AbstractArray)` now simply calls the underlying neural network on data. In other words, it returns the generic predictions, not LA predictions. This was implemented to facilitate better interplay with `MLJFlux`. [#39] 
- Moving straight to `1.0.0` now for package, because zero major versions cause compat headaches with other packages in Taija ecosystem. [#39]
- Removed support for `v1.7`, now `v1.9` as lower bound. This is because we are now overloading the `MLJFlux.train` and `MLJFlux.train_epoch` functions, which were added in version `v0.5.0` of that package, which is lower-bounded at `v1.9`. [#39]
- Updated codecov workflow in CI.yml. [#39]
- fixed test functions [#39]
- adapted the LaplaceClassification and the LaplaceRegression struct to use the new @mlj_model macro from MLJBase.[#39]
- Changed the fit! method arguments. [#39]
- Changed the predict functions for both LaplaceClassification and LaplaceRegression.[#39]

### Removed

- Removed the shape, build and clean! functions.[#39]
- Removed Review dog for code format suggestions. [#39]

### Added

- Added new keyword parameter ret_distr::Bool=false to predict. [#99]

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


>>>>>>> main

## Version [0.2.1] - 2024-05-29

### Changed

- Improved the docstring for the `predict` and `glm_predictive_distribution` methods. [#88]

### Added

- Added `probit` helper function to compute probit approximation for classification. [#88]
