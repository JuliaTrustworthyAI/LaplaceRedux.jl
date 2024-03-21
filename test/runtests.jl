using LaplaceRedux
using Test

@testset "LaplaceRedux.jl" begin

    # Quality assurance:
    include("aqua.jl")

    @testset "Data" begin
        include("data.jl")
    end

    @testset "Curvature" begin
        include("curvature.jl")
    end

    @testset "Laplace" begin
        include("laplace.jl")
    end

    if VERSION >= v"1.8.0"
        @testset "PyTorch Comparisons" begin
            include("pytorch_comparison.jl")
        end
    end

    @testset "Kron" begin
        include("kron.jl")
    end

    @testset "KronDecomposed" begin
        include("krondecomposed.jl")
    end

    @testset "MLJFlux" begin
        include("mlj_flux_interfacing.jl")
    end

    @testset "CounterfactualExplanations" begin
        include("counterfactual_explanations.jl")
    end
end
