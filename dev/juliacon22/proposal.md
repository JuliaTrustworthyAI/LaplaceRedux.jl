# Abstract

Treating deep neural networks probabilistically comes with numerous advantages including improved robustness and greater interpretability. These factors are key to building artificial intelligence (AI) that is trustworthy. A drawback commonly associated with existing Bayesian methods is that they increase computational costs. Recent work has shown that Bayesian deep learning can be effortless through Laplace approximation. This talk presents an implementation in Julia: `BayesLaplace.jl`.

# Description

#### Problem: Bayes can costly 😥

Deep learning models are typically heavily under-specified in the data, which makes them vulnerable to adversarial attacks and impedes interpretability. Bayesian deep learning promises an intuitive remedy: instead of relying on a single explanation for the data, we are interested in computing averages over many compelling explanations. Multiple approaches to Bayesian deep learning have been put forward in recent years including variational inference, deep ensembles and Monte Carlo dropout. Despite their usefulness these approaches involve additional computational costs compared to training just a single network. Recently, another promising approach has entered the limelight: Laplace approximation (LA).

#### Solution: Laplace Redux 🤩

While LA was first proposed in the 18th century, it has so far not attracted serious attention from the deep learning community largely because it involves a possibly large Hessian computation. The authors of this recent [NeurIPS paper](https://arxiv.org/abs/2106.14806) are on a mission to change the perception that LA has no use in DL: they demonstrate empirically that LA can be used to produce Bayesian model averages that are at least at par with existing approaches in terms of uncertainty quantification and out-of-distribution detection, while being significantly cheaper to compute. Our package [`BayesLaplace.jl`](https://github.com/pat-alt/BayesLaplace.jl) provides a light-weight implementation of this approach in Julia that allows users to recover Bayesian representations of deep neural networks in an efficient post-hoc manner.

#### Limitations and Goals 🚩

The package functionality is still limited to binary classification models trained in Flux. It also lacks any framework for optimizing with respect to the Bayesian prior. In future work we aim to extend the functionality. We would like to develop a library that is at least at par with an existing Python library: [Laplace](https://aleximmer.github.io/Laplace/). Contrary to the existing Python library, we would like to leverage Julia's support for language interoperability to also facilitate applications to deep neural networks trained in other programming languages like Python an R. 

#### Further reading 📚

For more information on this topic please feel free to check out this [introductory post](https://www.paltmeyer.com/blog/posts/effortsless-bayesian-dl/).

## Notes 

While this topic should be highly relevant to the Julia community, the package is still in its very early stages. If the quality at this point is not up to the standard of a lighting talk submission, I would also be very happy to present this as an experience talk.