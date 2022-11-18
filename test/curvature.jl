using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux


@testset "Jacobian" begin

    # One layer:
    nn = Chain(Dense(2,1))
    la = Laplace(nn; likelihood=:classification)

end

@testset "Hessian" begin
    
end