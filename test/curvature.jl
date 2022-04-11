using BayesLaplace
using BayesLaplace.Curvature
using BayesLaplace.Data
using Flux


@testset "Jacobian" begin

    # One layer:
    nn = Chain(Dense(2,1))
    la = Laplace(nn)

end

@testset "Hessian" begin
    
end