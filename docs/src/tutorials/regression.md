
``` @meta
CurrentModule = LaplaceRedux
```

## Data

We first generate some synthetic data:

``` julia
using LaplaceRedux.Data
fun(x) = sin(2*π*x)
n = 100     # number of observations
σtrue = 0.3       # true observational noise
x, y = Data.toy_data_regression(100;noise=σtrue,fun=fun)
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
    Dense(D, n_hidden, tanh_fast),
    Dense(n_hidden, n_hidden, tanh_fast),
    Dense(n_hidden, 1)
)  
λ = 0.01
sqnorm(x) = sum(abs2, x)
weight_regularization(λ=λ) = 1/2 * λ^2 * sum(sqnorm, Flux.params(nn))
loss(x, y) = Flux.Losses.mse(nn(x), y) + weight_regularization();
```

We train the model:

``` julia
using Flux.Optimise: update!, Adam
opt = Adam()
epochs = 100
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
la = Laplace(nn; likelihood=:regression, λ=λ, subset_of_weights=:last_layer, σ=σtrue)
fit!(la, data)
plot(la, X, y)
```

![](regression_files/figure-commonmark/cell-6-output-1.svg)