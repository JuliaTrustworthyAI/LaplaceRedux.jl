using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux

@testset "Construction" begin

    # One layer:
    nn = Chain(Dense(2,1))
    la = Laplace(nn; likelihood=:classification)
    @test la.n_params == 3

    # Multi-layer:
    nn = Chain(Dense(2,2,σ),Dense(2,1))
    la = Laplace(nn; likelihood=:regression, subset_of_weights=:last_layer)
    @test la.n_params == 3
    la = Laplace(nn; likelihood=:classification, subset_of_weights=:all)
    @test la.n_params == 9

end

@testset "Parameters" begin
    
    nn = Chain(Dense(2,2,σ),Dense(2,1))
    la = Laplace(nn; likelihood=:classification, subset_of_weights=:last_layer)
    @test LaplaceRedux.get_params(la) == collect(Flux.params(nn))[(end-1):end]

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
        x = [1,1]
        grad = [-0.5,-0.5] # analytical solution for gradient
        @test LaplaceRedux.hessian_approximation(la, (x,target))[1:2,1:2] == grad * grad'

        x = [-1,-1]
        grad = [0.5,0.5] # analytical solution for gradient
        @test LaplaceRedux.hessian_approximation(la, (x,target))[1:2,1:2] == grad * grad'

        x = [0,0]
        grad = [0,0] # analytical solution for gradient
        @test LaplaceRedux.hessian_approximation(la, (x,target))[1:2,1:2] == grad * grad'
        
    end
    
end


# We know the analytical expression for the Hessian of logit binary cross entropy loss for a single-layer neural net with sigmoid activation just corresponds to the Hessian in logistic regression (see for example: https://www.paltmeyer.com/blog/posts/bayesian-logit/): ∇ℓ=(μ-y)(μ(1-μ)xx'). With a weight-penalty (Gaussian prior), the Hessian becomes: ∇ℓ=∑(μ-y)(μ(1-μ)xx')+H₀. We can use this analytical expression to see if we get the expected results.

@testset "Fitting" begin

    nn = Chain(Dense([0 0]))
    la = Laplace(nn; likelihood=:classification)

    hessian_exact(x,target) = (nn(x).-target).*(nn(x).*(1 .- nn(x)).*x*x') + la.H₀

    @testset "Empirical Fisher - full" begin
        
        target = [1]
        x = [[0,0]]
        fit!(la,zip(x,target))
        @test la.H[1:2,1:2] == hessian_exact(x[1],target[1])

    end

end


@testset "Complete Workflow" begin
    
    using Flux
    using Flux.Optimise: update!, Adam
    using Statistics

    # Number of points to generate.
    xs, ys = LaplaceRedux.Data.toy_data_non_linear(200)
    X = hcat(xs...) # bring into tabular format
    data = zip(xs,ys)

    # Neural network:
    n_hidden = 32
    D = size(X)[1]
    nn = Chain(
        Dense(D, n_hidden, σ),
        Dense(n_hidden, 1)
    )  
    λ = 0.01
    sqnorm(x) = sum(abs2, x)
    weight_regularization(λ=λ) = 1/2 * λ^2 * sum(sqnorm, Flux.params(nn))
    loss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y) + weight_regularization()

    opt = Adam()
    epochs = 200
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

    la = Laplace(nn; likelihood=:classification, λ=λ, subset_of_weights=:last_layer)
    fit!(la, data)

    p̂ = predict(la, X)
    @test isa(p̂, Matrix)

end