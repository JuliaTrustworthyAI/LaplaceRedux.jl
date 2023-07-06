# Data

``` @meta
CurrentModule = LaplaceRedux
```

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

Next we optimize the prior precision *P*₀ and and observational noise *σ* using Empirical Bayes:

``` julia
optimize_prior!(la; verbose=true)
plot(la, X, y; zoom=-5, size=(400,400))
```

    loss(exp.(logP₀), exp.(logσ)) = 115.75735421748533
    Log likelihood: -79.2200866848734
    Log det ratio: 41.447703478591464
    Scatter: 31.626831586632402
    loss(exp.(logP₀), exp.(logσ)) = 117.86642880183206
    Log likelihood: -86.56623967031658
    Log det ratio: 47.038810369756135
    Scatter: 15.561567893274834
    loss(exp.(logP₀), exp.(logσ)) = 99.47631633578109
    Log likelihood: -69.09330443512576
    Log det ratio: 50.58494846478561
    Scatter: 10.18107533652504
    loss(exp.(logP₀), exp.(logσ)) = 98.39016022405285
    Log likelihood: -68.06679631670653
    Log det ratio: 52.35553006863444
    Scatter: 8.291197746058197
    loss(exp.(logP₀), exp.(logσ)) = 98.32796211733199
    Log likelihood: -67.97958599228153
    Log det ratio: 52.957723488965314
    Scatter: 7.739028761135608
    loss(exp.(logP₀), exp.(logσ)) = 97.4846502485071
    Log likelihood: -67.13783329716615
    Log det ratio: 52.93077798129934
    Scatter: 7.7628559213825294
    loss(exp.(logP₀), exp.(logσ)) = 97.35181187147037
    Log likelihood: -67.01886503956803
    Log det ratio: 52.64896810867424
    Scatter: 8.016925555130463
    loss(exp.(logP₀), exp.(logσ)) = 97.36438554054067
    Log likelihood: -67.04148717903837
    Log det ratio: 52.335624298852
    Scatter: 8.310172424152613
    loss(exp.(logP₀), exp.(logσ)) = 97.349858431591
    Log likelihood: -67.03065735188642
    Log det ratio: 52.098394400933444
    Scatter: 8.540007758475733
    loss(exp.(logP₀), exp.(logσ)) = 97.33749417115928
    Log likelihood: -67.01884461669445
    Log det ratio: 51.96485166001452
    Scatter: 8.67244744891513

![](regression_files/figure-commonmark/cell-7-output-2.svg)
