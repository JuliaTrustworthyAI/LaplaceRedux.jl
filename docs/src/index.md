```@meta
CurrentModule = BayesLaplace
```

# BayesLaplace

Documentation for [BayesLaplace](https://github.com/pat-alt/BayesLaplace.jl).

This is a small library that can be used for effortless Bayesian Deep Learning and Logisitic Regression trough Laplace Approximation. It is inspired by this Python [library](https://aleximmer.github.io/Laplace/index.html#setup) and its companion [paper](https://arxiv.org/abs/2106.14806).

## Installation

This package is not registered, but can be installed from Github as follows:

```julia
using Pkg
Pkg.add("https://github.com/pat-alt/BayesLaplace.jl")
```

## Limitations

This library is pure-play and lacks any kind of unit testing. It is also limited to binary classification problems. 