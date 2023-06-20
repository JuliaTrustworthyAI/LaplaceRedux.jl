using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux
using Flux.Optimise: update!, Adam
using Plots
using Statistics
using MLUtils
using LinearAlgebra

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

    # (always ignoring constant)
    @testset "Empirical Fisher - full" begin
        la = Laplace(nn; likelihood=:classification, backend=:EmpiricalFisher)

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

        # Regression
        la = Laplace(nn; likelihood=:regression, backend=:EmpiricalFisher)
        target = 3
        x = [1, 2]
        grad = [-6.0, -12.0, -6.0]
        _, H = LaplaceRedux.hessian_approximation(la, (x, target))
        @test H == grad * grad'
    end

    @testset "Generalized Gauss–Newton (GGN) - full" begin
        # Regression
        la = Laplace(nn; likelihood=:regression, backend=:GGN)
        target = 3
        x = [1, 2]
        J = [1; 2; 1;;] # pre-calculated jacobians
        _, H = LaplaceRedux.hessian_approximation(la, (x, target))
        @test H == J * J'

        # Binary Classification
        la = Laplace(nn; likelihood=:classification, backend=:GGN)
        target = 1
        x = [1, 1]
        J = [1; 1; 1;;] # pre-calculated jacobians
        f_mu = [0]
        p = sigmoid(f_mu)
        H_lik = diagm(p) - p * p'
        _, H = LaplaceRedux.hessian_approximation(la, (x, target))
        @test H == J * H_lik * J'
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

function train_nn(val::Dict; verbose=false)
    # Unpack:
    X = val[:X]
    Y = val[:Y]
    # NOTE: for classification multi, Y is one-hot, y is labels as itegers (1-N)
    y = val[:y]
    data = val[:data]
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

    return nn
end

function run_workflow(
    val::Dict,
    batchsize::Int,
    backend::Symbol,
    subset_of_weights::Symbol;
    hessian_structure=:full,
    verbose::Bool=false,
    do_optimize_prior::Bool=true,
    do_predict::Bool=true,
    do_plot::Bool=true,
)
    # Unpack:
    X = val[:X]
    Y = val[:Y]
    # NOTE: for classification multi, Y is one-hot, y is labels as itegers (1-N)
    y = val[:y]
    if batchsize == 0
        data = val[:data]
    else
        data = DataLoader((X, Y); batchsize=batchsize)
    end
    outdim = val[:outdim]
    loss_fun = val[:loss_fun]
    likelihood = val[:likelihood]

    nn = val[:nn]
    λ = 0.01

    # NOTE: this hardcoded parameter value only matters in case `subset_of_weights==:subnet`
    # I will work for all D => 32 => K networks.
    subnetwork_indices = [[1, 1, 1], [2, 1], [2, 2], [3, 1, 16], [4, 1]]

    la = Laplace(
        nn;
        likelihood=likelihood,
        λ=λ,
        subset_of_weights=subset_of_weights,
        backend=backend,
        subnetwork_indices=subnetwork_indices,
        hessian_structure=hessian_structure,
    )
    fit!(la, data)
    if do_optimize_prior
        optimize_prior!(la; verbose=verbose)
    end
    if do_predict
        predict(la, X)
    end
    if do_plot
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
                    plt = plot(
                        la, X, y; target=target, clim=(0, 1), link_approx=link_approx
                    )
                    push!(plt_list, plt)
                end
                plot(plt_list...)
            end
        end
    end

    return la.H
end

@testset "Complete Workflows" begin

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
    batchsizes = [0, 32]
    backends = [:GGN, :EmpiricalFisher]
    subsets_of_weights = [:all, :last_layer, :subnetwork]
    # NOTE: not used yet
    hessian_structures = [:full, :kron]

    # Store Hessians to compare them for different batchsizes
    hessians = Dict()

    println("Training networks.")
    for (likelihood, val) in data_dict
        val[:nn] = train_nn(val)
    end

    println("Running workflows.")
    for (likelihood, val) in data_dict
        for (backend, batchsize, subset_of_weights) in
            Iterators.product(backends, batchsizes, subsets_of_weights)
            batchsize_text = batchsize == 0 ? "unbatched" : "batchsize=$(batchsize)"
            @testset "$(likelihood), $(batchsize_text), backend=$(backend), subset_of_weights=$(subset_of_weights)" begin
                println((likelihood, batchsize, backend, subset_of_weights))
                hessians[likelihood, batchsize, backend, subset_of_weights] = run_workflow(
                    val, batchsize, backend, subset_of_weights
                )
            end
        end
    end

    # Compare collected Hessians
    @testset "Comparing Hessians for varying batchsizes" begin
        for ((likelihood, val), backend, subset_of_weights) in
            Iterators.product(data_dict, backends, subsets_of_weights)
            println((likelihood, backend, subset_of_weights))
            hessians_by_batch = [
                hessians[likelihood, batchsize, backend, subset_of_weights] for
                batchsize in batchsizes
            ]
            # Compare consecutive Hessians
            for (H_i, H_j) in zip(hessians_by_batch, hessians_by_batch[2:end])
                @test H_i ≈ H_j atol = 0.05
            end
        end
    end

    # Support for KFAC limited, hence the separate test
    @testset "KFAC, unbatched, classification_multi, Fisher" begin
        likelihood = :classification_multi
        val = data_dict[likelihood]
        K = run_workflow(
            val,
            0,
            :EmpiricalFisher,
            :all;
            hessian_structure=:kron,
            do_optimize_prior=false,
            do_predict=true,
            do_plot=false,
        )
    end
end
