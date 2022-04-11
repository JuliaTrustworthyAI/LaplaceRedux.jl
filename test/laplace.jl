using BayesLaplace
using BayesLaplace.Curvature
using BayesLaplace.Data
using Flux

@testset "Construction" begin

    # One layer:
    nn = Chain(Dense(2,1))
    la = Laplace(nn)
    @test la.n_params == 3

    # Multi-layer:
    nn = Chain(Dense(2,2,σ),Dense(2,1))
    la = Laplace(nn)
    @test la.n_params == 3
    la = Laplace(nn; subset_of_weights=:all)
    @test la.n_params == 9

end

@testset "Parameters" begin
    
    nn = Chain(Dense(2,2,σ),Dense(2,1))
    la = Laplace(nn)
    @test BayesLaplace.get_params(la) == collect(Flux.params(nn))[(end-1):end]

    la = Laplace(nn; subset_of_weights=:all)
    @test BayesLaplace.get_params(la) == collect(Flux.params(nn))

end


# We know the analytical expression for the gradient of logit binary cross entropy loss for a single-layer neural net with sigmoid activation just corresponds to the gradient in logistic regression (see for example: https://www.paltmeyer.com/blog/posts/bayesian-logit/): ∇ℓ=(μ-y)x. We can use this analytical expression to see if we get the expected results.

@testset "Hessian" begin
    
    nn = Chain(Dense([0 0]))
    la = Laplace(nn)

    # (always ignoring constant)
    @testset "Empirical Fisher - full" begin
       
        target = 1
        x = [1,1]
        grad = [-0.5,-0.5] # analytical solution for gradient
        @test BayesLaplace.hessian_approximation(la, (x,target))[1:2,1:2] == grad * grad'

        x = [-1,-1]
        grad = [0.5,0.5] # analytical solution for gradient
        @test BayesLaplace.hessian_approximation(la, (x,target))[1:2,1:2] == grad * grad'

        x = [0,0]
        grad = [0,0] # analytical solution for gradient
        @test BayesLaplace.hessian_approximation(la, (x,target))[1:2,1:2] == grad * grad'
        
    end
    
end


# We know the analytical expression for the Hessian of logit binary cross entropy loss for a single-layer neural net with sigmoid activation just corresponds to the Hessian in logistic regression (see for example: https://www.paltmeyer.com/blog/posts/bayesian-logit/): ∇ℓ=(μ-y)(μ(1-μ)xx'). With a weight-penalty (Gaussian prior), the Hessian becomes: ∇ℓ=∑(μ-y)(μ(1-μ)xx')+H₀. We can use this analytical expression to see if we get the expected results.

@testset "Fitting" begin

    nn = Chain(Dense([0 0]))
    la = Laplace(nn)

    hessian_exact(x,target) = (nn(x).-target).*(nn(x).*(1 .- nn(x)).*x*x') + la.H₀

    @testset "Empirical Fisher - full" begin
        
        target = [1]
        x = [[0,0]]
        fit!(la,zip(x,target))
        @test la.H[1:2,1:2] == hessian_exact(x[1],target[1])

    end

end