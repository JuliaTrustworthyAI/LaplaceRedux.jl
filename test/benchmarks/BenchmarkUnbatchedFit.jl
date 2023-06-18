using Pkg
Pkg.activate(".")
using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux
using Flux.Optimise: update!, Adam
using Statistics
using MLUtils
using Zygote
using Printf
using BenchmarkTools
using Tullio

# Setup some dummy data, like in tests

n = 128 * 128
data_dict = Dict()
bsize = 2

x, y = LaplaceRedux.Data.toy_data_regression(n)
xs = [[x] for x in x]
X, Y = reduce(hcat, x), reduce(hcat, y)

dataloader = DataLoader((X, Y), batchsize=bsize)
data = zip(xs, y)
data_dict[:regression] = Dict(
    :data => data,
    :X => X,
    :y => y,
    :outdim => 1,
    :loss_fun => :mse,
    :likelihood => :regression,
)

# Train a NN model

val = data_dict[:regression]

# Unpack:
data = val[:data]
X = val[:X]
y = val[:y]
outdim = val[:outdim]
loss_fun = val[:loss_fun]
likelihood = val[:likelihood]

# Neural network:
n_hidden = 32
D = size(X, 1)
nn = Chain(Dense(D, n_hidden, σ), Dense(n_hidden, outdim))
λ = 0.01
sqnorm(x) = sum(abs2, x)
weight_regularization(λ=λ) = 1 / 2 * λ^2 * sum(sqnorm, Flux.params(nn))
loss(x, y) = getfield(Flux.Losses, loss_fun)(nn(x), y) + weight_regularization()


opt = Adam()
epochs = 20#0
avg_loss(data) = mean(map(d -> loss(d[1], d[2]), data))
show_every = epochs / 10

for epoch in 1:epochs
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

function fit_la_unbatched(nn, data, X, y)
    la = Laplace(nn; likelihood=:regression, λ=λ, subset_of_weights=:all)
    fit!(la, data)
end

suite = BenchmarkGroup()

suite["fit_la_unbatched"] = BenchmarkGroup(["tag1", "tag2"])

suite["fit_la_unbatched"][1]= @benchmarkable fit_la_unbatched($nn, $data, $X, $y) 
tune!(suite)
results = run(suite, verbose = true)

BenchmarkTools.save("   output.json", median(results))
