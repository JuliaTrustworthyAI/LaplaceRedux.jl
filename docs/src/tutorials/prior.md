

``` @meta
CurrentModule = LaplaceRedux
```

!!! note "In Progress"  
    This documentation is still incomplete.

## A quick note on the prior

### General Effect

High prior precision $\rightarrow$ only observation noise. Low prior precision $\rightarrow$ high posterior uncertainty.

``` julia
using LaplaceRedux.Data
n = 150       # number of observations
σtrue = 0.30  # true observational noise
x, y = Data.toy_data_regression(n;noise=σtrue)
xs = [[x] for x in x]
X = permutedims(x)
```

![](prior_files/figure-commonmark/cell-4-output-1.svg)

### Effect of Model Size on Optimal Choice

For larger models, the optimal prior precision $\lambda$ as evaluated through Empirical Bayes tends to be smaller.

![](prior_files/figure-commonmark/cell-5-output-1.svg)

![](prior_files/figure-commonmark/cell-6-output-1.svg)
