using BayesLaplace
using Test

@testset "BayesLaplace.jl" begin
    @testset "Curvature" begin
        include("curvature.jl")
    end

    @testset "Laplace" begin
        include("laplace.jl")
    end
end
