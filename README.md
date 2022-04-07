# BayesLaplace

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/dev) [![Build Status](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl)

This is a small library that can be used for effortless Bayesian Deep Learning and Logisitic Regression trough Laplace Approximation. It is inspired by this Python [library](https://aleximmer.github.io/Laplace/index.html#setup) and its companion [paper](https://arxiv.org/abs/2106.14806).

## Installation

This package is not registered, but can be installed from Github as follows:

``` julia
using Pkg
Pkg.add("https://github.com/pat-alt/BayesLaplace.jl")
```

## Getting started

Laplace approximation can be used post-hoc for any trained neural network. This library should be compatible with any pre-trained Flux.jl model. Let `nn` be one such model trained on dataset `data`. Then implementing Laplace approximation is easy as follows:

``` julia
la = laplace(nn)
fit!(la, data)
```

Calling `predict(nn,X)` for some features `X` will produce posterior predictions. The plot below has been lifted from the [documentation](https://www.paltmeyer.com/BayesLaplace.jl/dev/), which provides more detail. It shows the resulting posterior predictive surface for the plugin estimator (left) and the Laplace approximation (right) for a toy data set.

![](https://raw.githubusercontent.com/pat-alt/BayesLaplace.jl/main/docs/src/tutorials/www/posterior_predictive_mlp.png)

## Limitations

This library is pure-play and lacks any kind of unit testing and documenation of functions and classes. It is also limited to binary classification problems.
