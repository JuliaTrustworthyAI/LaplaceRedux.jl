using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux
using Flux.Optimise: update!, Adam
using Plots
using Statistics
using MLUtils

@testset "Construction" begin

    # One layer:
    nn = Chain(Dense(2, 1))

    # Expected error
    @test_throws AssertionError Laplace(
        nn; likelihood=:classification, subset_of_weights=:last
    )

    # Correct:
    la = Laplace(nn; likelihood=:classification)
    @test la.n_params == 3

    # Multi-layer:
    nn = Chain(Dense(2, 2, σ), Dense(2, 1))
    la = Laplace(nn; likelihood=:regression, subset_of_weights=:last_layer)
    @test la.n_params == 3
    la = Laplace(nn; likelihood=:classification, subset_of_weights=:all)
    @test la.n_params == 9
end

@testset "Parameters" begin
    nn = Chain(Dense(2, 2, σ), Dense(2, 1))
    la = Laplace(nn; likelihood=:classification, subset_of_weights=:last_layer)
    @test LaplaceRedux.get_params(la) == collect(Flux.params(nn))[(end - 1):end]

    la = Laplace(nn; likelihood=:classification, subset_of_weights=:all)
    @test LaplaceRedux.get_params(la) == collect(Flux.params(nn))
end

# We know the analytical expression for the gradient of logit binary cross entropy loss for a single-layer neural net with sigmoid activation just corresponds to the gradient in logistic regression (see for example: https://www.paltmeyer.com/blog/posts/bayesian-logit/): ∇ℓ=(μ-y)x. We can use this analytical expression to see if we get the expected results.

@testset "Hessian" begin
    nn = Chain(Dense([0 0]))
    la = Laplace(nn; likelihood=:classification)

    # (always ignoring constant)
    @testset "Empirical Fisher - full" begin
        target = 1
        x = [1, 1]
        grad = [-0.5, -0.5] # analytical solution for gradient
        _, H = LaplaceRedux.hessian_approximation(la, (x, target))
        @test H[1:2, 1:2] == grad * grad'

        x = [-1, -1]
        grad = [0.5, 0.5] # analytical solution for gradient
        _, H = LaplaceRedux.hessian_approximation(la, (x, target))
        @test H[1:2, 1:2] == grad * grad'

        x = [0, 0]
        grad = [0, 0] # analytical solution for gradient
        _, H = LaplaceRedux.hessian_approximation(la, (x, target))
        @test H[1:2, 1:2] == grad * grad'
    end
end

# We know the analytical expression for the Hessian of logit binary cross entropy loss for a single-layer neural net with sigmoid activation just corresponds to the Hessian in logistic regression (see for example: https://www.paltmeyer.com/blog/posts/bayesian-logit/): ∇ℓ=(μ-y)(μ(1-μ)xx'). With a weight-penalty (Gaussian prior), the Hessian becomes: ∇ℓ=∑(μ-y)(μ(1-μ)xx')+P₀. We can use this analytical expression to see if we get the expected results.

@testset "Fitting" begin
    nn = Chain(Dense([0 0]))
    la = Laplace(nn; likelihood=:classification)

    function hessian_exact(x, target)
        return (nn(x) .- target) .* (nn(x) .* (1 .- nn(x)) .* x * x') + la.P₀[1:2, 1:2]
    end

    @testset "Empirical Fisher - full" begin
        target = [1]
        x = [[0, 0]]
        fit!(la, zip(x, target))
        @test la.P[1:2, 1:2] == hessian_exact(x[1], target[1])
    end
end

function run_workflow(val::Dict, batchsize::Int, backend::Symbol, subset_of_weights::Symbol; verbose::Bool=false)
    # Unpack:
    X = val[:X]
    Y = val[:Y]
    # NOTE: for classification multi, Y is one-hot, y is labels as itegers (1-N)
    y = val[:y]
    if batchsize == 0
        data = val[:data]
    else
        data = DataLoader((X, Y), batchsize=batchsize)
    end
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
    epochs = 200
    avg_loss(data) = mean(map(d -> loss(d[1], d[2]), data))
    show_every = epochs / 10

    for epoch in 1:epochs
        for d in data
            gs = gradient(Flux.params(nn)) do
                l = loss(d...)
            end
            update!(opt, Flux.params(nn), gs)
        end
        if verbose && epoch % show_every == 0
            println("Epoch " * string(epoch))
            @show avg_loss(data)
        end
    end

    la = Laplace(nn; likelihood=likelihood, λ=λ, subset_of_weights=subset_of_weights, backend=backend)
    fit!(la, data)
    optimize_prior!(la; verbose=verbose)
    if outdim == 1
        plot(la, X, y)                              # standard
        plot(la, X, y; xlims=(-5, 5), ylims=(-5, 5))  # lims
        plot(la, X, y; link_approx=:plugin)         # plugin approximation  
    else
        # Classification multi is plotted differently
        @assert likelihood == :classification
        # NOTE: do we not allow for vector-output regression?
        for link_approx in [:probit, :plugin]
            _labels = sort(unique(y))
            plt_list = []
            for target in _labels
                plt = plot(la, X, y; target=target, clim=(0,1), link_approx=link_approx)
                push!(plt_list, plt)
            end
            plot(plt_list...)
        end
    end
end

@testset "Complete Workflow" begin

    # SETUP
    n = 100
    data_dict = Dict()

    # Classification binary:
    xs, y = LaplaceRedux.Data.toy_data_non_linear(n)
    X = reduce(hcat, xs)
    Y = reduce(hcat, y)
    data = zip(xs, y)
    data_dict[:classification_binary] = Dict(
        :data => data,
        :X => X,
        :Y => Y,
        :y => y,
        :outdim => 1,
        :loss_fun => :logitbinarycrossentropy,
        :likelihood => :classification,
    )

    # Classification multi:
    xs, y = LaplaceRedux.Data.toy_data_multi(n)
    ytrain = Flux.onehotbatch(y, unique(y))
    ytrain = Flux.unstack(ytrain'; dims=1)
    X = reduce(hcat, xs)
    Y = reduce(hcat, ytrain)
    data = zip(xs, ytrain)
    data_dict[:classification_multi] = Dict(
        :data => data,
        :X => X,
        :Y => Y,
        :y => y,
        :outdim => length(first(ytrain)),
        :loss_fun => :logitcrossentropy,
        :likelihood => :classification,
    )

    # Regression:
    x, y = LaplaceRedux.Data.toy_data_regression(n)
    xs = [[x] for x in x]
    X = reduce(hcat, xs)
    Y = reduce(hcat, y)
    data = zip(xs, y)
    data_dict[:regression] = Dict(
        :data => data,
        :X => X,
        :Y => Y,
        :y => y,
        :outdim => 1,
        :loss_fun => :mse,
        :likelihood => :regression,
    )

    # WORKFLOWS

    # NOTE: batchsize=0 is meant to represent unbatched
    batchsizes = [0, 1, 32]
    backends = [:GGN, :EmpiricalFisher]
    weight_subsets = [:all, :last_layer]
    # TODO add subnet

    for ((likelihood, val), backend, batchsize, weight_subset) in Iterators.product(data_dict, backends, batchsizes, weight_subsets)
        batchsize_text = batchsize == 0 ? "unbatched" : "batchsize=$(batchsize)"
        @testset "$(likelihood), $(batchsize_text), backend=$(backend), subset_of_weights=$(weight_subset)" begin
            run_workflow(val, batchsize, backend, weight_subset)
        end
    end
end