using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux
using Flux.Optimise: update!, Adam
using Plots
using Statistics

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
    @test_throws AssertionError Laplace(
        nn; likelihood=:classification, subset_of_weights=:subnetwork
    )
    @test_throws AssertionError Laplace(
        nn;
        likelihood=:classification,
        subset_of_weights=:subnetwork,
        subnetwork_indices=[[1, 1, 1], [3, 1, 1], [5, 1]],
    )
    @test_throws AssertionError Laplace(
        nn;
        likelihood=:classification,
        subset_of_weights=:subnetwork,
        subnetwork_indices=[[1, 1, 1], [6, 1, 1], [4, 1]],
    )
    @test_throws AssertionError Laplace(
        nn;
        likelihood=:classification,
        subset_of_weights=:subnetwork,
        subnetwork_indices=[[1, 1, 1, 1], [6, 1, 1], [4, 1]],
    )
    la = Laplace(
        nn;
        likelihood=:classification,
        subset_of_weights=:subnetwork,
        subnetwork_indices=[[1, 1, 1], [3, 1, 1], [4, 1]],
    )
    @test la.n_params == 3
    @test la.curvature.subnetwork_indices == [1, 7, 9]

    # Testing index conversion for all weights:
    nn = Chain(Dense(2, 10, σ), Dense(10, 1))
    la = Laplace(
        nn;
        likelihood=:classification,
        subset_of_weights=:subnetwork,
        subnetwork_indices=[
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
            [1, 3, 1],
            [1, 3, 2],
            [1, 4, 1],
            [1, 4, 2],
            [1, 5, 1],
            [1, 5, 2],
            [1, 6, 1],
            [1, 6, 2],
            [1, 7, 1],
            [1, 7, 2],
            [1, 8, 1],
            [1, 8, 2],
            [1, 9, 1],
            [1, 9, 2],
            [1, 10, 1],
            [1, 10, 2],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [2, 7],
            [2, 8],
            [2, 9],
            [2, 10],
            [3, 1, 1],
            [3, 1, 2],
            [3, 1, 3],
            [3, 1, 4],
            [3, 1, 5],
            [3, 1, 6],
            [3, 1, 7],
            [3, 1, 8],
            [3, 1, 9],
            [3, 1, 10],
            [4, 1],
        ],
    )
    println(la.curvature.subnetwork_indices)
    @test la.n_params == 41
    @test la.curvature.subnetwork_indices == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
    ]
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

@testset "Complete Workflow" begin

    # SETUP
    n = 100
    data_dict = Dict()

    # Classification binary:
    xs, y = LaplaceRedux.Data.toy_data_non_linear(n)
    X = hcat(xs...)
    data = zip(xs, y)
    data_dict[:classification_binary] = Dict(
        :data => data,
        :X => X,
        :y => y,
        :outdim => 1,
        :loss_fun => :logitbinarycrossentropy,
        :likelihood => :classification,
    )

    # Classification Multi:
    xs, y = LaplaceRedux.Data.toy_data_multi(n)
    X = hcat(xs...)
    ytrain = Flux.onehotbatch(y, unique(y))
    ytrain = Flux.unstack(ytrain'; dims=1)
    data = zip(xs, ytrain)
    data_dict[:classification_multi] = Dict(
        :data => data,
        :X => X,
        :y => y,
        :outdim => length(first(ytrain)),
        :loss_fun => :logitcrossentropy,
        :likelihood => :classification,
    )

    # Regression:
    x, y = LaplaceRedux.Data.toy_data_regression(n)
    xs = [[x] for x in x]
    X = hcat(xs...)
    data = zip(xs, y)
    data_dict[:regression] = Dict(
        :data => data,
        :X => X,
        :y => y,
        :outdim => 1,
        :loss_fun => :mse,
        :likelihood => :regression,
    )

    # WORKFLOWS

    for (likelihood, val) in data_dict
        @testset "$(likelihood)" begin

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
                if epoch % show_every == 0
                    println("Epoch " * string(epoch))
                    @show avg_loss(data)
                end
            end

            if outdim == 1
                la = Laplace(nn; likelihood=likelihood, λ=λ, subset_of_weights=:last_layer)
                fit!(la, data)
                optimize_prior!(la; verbose=true)
                plot(la, X, y)                              # standard
                plot(la, X, y; xlims=(-5, 5), ylims=(-5, 5))  # lims
                plot(la, X, y; link_approx=:plugin)         # plugin approximation
                #else
                #    @test_throws AssertionError Laplace(
                #        nn; likelihood=likelihood, λ=λ, subset_of_weights=:last_layer
                #    )
                la = Laplace(
                    nn;
                    likelihood=likelihood,
                    λ=λ,
                    subset_of_weights=:subnetwork,
                    subnetwork_indices=[[1, 1, 1], [2, 1], [2, 2], [3, 1, 16], [4, 1]],
                )
                fit!(la, data)
                optimize_prior!(la; verbose=true)
                plot(la, X, y)                              # standard
                plot(la, X, y; xlims=(-5, 5), ylims=(-5, 5))  # lims
                plot(la, X, y; link_approx=:plugin)         # plugin approximation
            end
        end
    end
end
