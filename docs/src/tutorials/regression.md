

``` @meta
CurrentModule = LaplaceRedux
```

## Libraries

Import the libraries required to run this example

``` julia
using Pkg; Pkg.activate("docs")
# Import libraries
using Flux, Plots, TaijaPlotting, Random, Statistics, LaplaceRedux
theme(:wong)
```

## Data

We first generate some synthetic data:

``` julia
using LaplaceRedux.Data
n = 300       # number of observations
σtrue = 0.30  # true observational noise
x, y = Data.toy_data_regression(n;noise=σtrue)
xs = [[x] for x in x]
X = permutedims(x)
```

## MLP

We set up a model and loss with weight regularization:

``` julia
data = zip(xs,y)
n_hidden = 50
D = size(X,1)
nn = Chain(
    Dense(D, n_hidden, tanh),
    Dense(n_hidden, 1)
)  
loss(x, y) = Flux.Losses.mse(nn(x), y)
```

We train the model:

``` julia
using Flux.Optimise: update!, Adam
opt = Adam(1e-3)
epochs = 1000
avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))
show_every = epochs/10

for epoch = 1:epochs
  for d in data
    gs = gradient(Flux.params(nn)) do
      l = loss(d...)
    end
    update!(opt, Flux.params(nn), gs)
  end
  if epoch % show_every == 0
    println("Epoch " * string(epoch))
    @show avg_loss(data)
  end
end
```

## Laplace Approximation

Laplace approximation can be implemented as follows:

``` julia
subset_w = :all
la = Laplace(nn; likelihood=:regression, subset_of_weights=subset_w)
fit!(la, data)
plot(la, X, y; zoom=-5, size=(400,400))
```

![](regression_files/figure-commonmark/cell-6-output-1.svg)

Next we optimize the prior precision $P_0$ and and observational noise $\sigma$ using Empirical Bayes:

``` julia
optimize_prior!(la; verbose=true)
plot(la, X, y; zoom=-5, size=(400,400))
```

    loss(exp.(logP₀), exp.(logσ)) = 104.78561546028183
    Log likelihood: -70.48742092717352
    Log det ratio: 41.1390695290454
    Scatter: 27.45731953717124
    loss(exp.(logP₀), exp.(logσ)) = 104.9736282327825
    Log likelihood: -74.85481357633174
    Log det ratio: 46.59827618892447
    Scatter: 13.639353123977058
    loss(exp.(logP₀), exp.(logσ)) = 84.38222356291794
    Log likelihood: -54.86985627702764
    Log det ratio: 49.92347667032635
    Scatter: 9.101257901454279

    loss(exp.(logP₀), exp.(logσ)) = 84.53493863039972
    Log likelihood: -55.013137224636
    Log det ratio: 51.43622180356522
    Scatter: 7.607381007962245
    loss(exp.(logP₀), exp.(logσ)) = 83.95921598606084
    Log likelihood: -54.41492266831395
    Log det ratio: 51.794520967146354
    Scatter: 7.294065668347427
    loss(exp.(logP₀), exp.(logσ)) = 83.03505059021086
    Log likelihood: -53.50540374805591
    Log det ratio: 51.574749787874794
    Scatter: 7.484543896435117

    loss(exp.(logP₀), exp.(logσ)) = 82.97840036025443
    Log likelihood: -53.468475394115416
    Log det ratio: 51.17273666609066
    Scatter: 7.847113266187348
    loss(exp.(logP₀), exp.(logσ)) = 82.98550025321256
    Log likelihood: -53.48508828283467
    Log det ratio: 50.81442045868749
    Scatter: 8.186403482068298
    loss(exp.(logP₀), exp.(logσ)) = 82.9584040552644
    Log likelihood: -53.45989630330948
    Log det ratio: 50.59063282947659
    Scatter: 8.406382674433235


    loss(exp.(logP₀), exp.(logσ)) = 82.94465052328141
    Log likelihood: -53.44600301956443
    Log det ratio: 50.500079294094405
    Scatter: 8.497215713339543

![](regression_files/figure-commonmark/cell-7-output-5.svg)
