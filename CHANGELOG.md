# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

*Note*: We try to adhere to these practices as of version [v0.2.1].

## Version [0.3.1] - 2024-06-22

### Changed

- Changed `glm_predictive_distribution` so that return a tuple(Normal distribution,fμ, fvar) rather than the tuple (mean,variance). [#90]

## Version [0.3.0] - 2024-06-21

### Changed

- Changed `glm_predictive_distribution` so that return a Normal distribution rather than the tuple (mean,variance). [#90]
- Changed `predict` so that return directly a Normal distribution  in the case of regression. [#90]

### Added

- Added functions to compute the average empirical frequency for both classification and regression problems in utils.jl. [#90]





## Version [0.2.1] - 2024-05-29

### Changed

- Improved the docstring for the `predict` and `glm_predictive_distribution` methods. [#88]

### Added

- Added `probit` helper function to compute probit approximation for classification. [#88]