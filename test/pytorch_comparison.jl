using LaplaceRedux
using CSV
using Test
using Flux
using Serialization
using DataFrames
include("testutils.jl")

# This test suite compares results for the hessian between the current Julia package and the corresponding Laplace from PyTorch as reference.
# https://github.com/AlexImmer/Laplace

@testset "Pytorch Hessian Comparisons" begin
    @testset "Multi-Class Classification" begin

        # Read the dataset from a CSV file
        df = CSV.read(joinpath(@__DIR__, "datafiles", "data_multi.csv"), DataFrame)
        x = Matrix(df[:, 1:2])
        x = [x[i, :] for i in 1:size(x, 1)]
        y = df[:, 3]

        X = hcat(x...)
        y_train = Flux.onehotbatch(y, unique(y))
        y_train = Flux.unstack(y_train', 1)

        data = zip(x, y_train)

        # Read the network weights and biases from a JLB file
        nn = deserialize(joinpath(@__DIR__, "datafiles", "nn-binary_multi.jlb"))

        @testset "LA - full weights - full hessian - ggn" begin
            la = Laplace(
                nn;
                likelihood=:classification,
                hessian_structure=:full,
                subset_of_weights=:all,
                backend=:GGN,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_multi_all_full_ggn")
            @test isapprox(pytorch_hessian, rearrange_hessian(la.H, nn); atol=0.0001)
        end

        @testset "LA - full weights - full hessian - empfisher" begin
            la = Laplace(
                nn;
                likelihood=:classification,
                hessian_structure=:full,
                subset_of_weights=:all,
                backend=:EmpiricalFisher,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_multi_all_full_empfisher")
            @test isapprox(pytorch_hessian, rearrange_hessian(la.H, nn); atol=0.0001)
        end

        @testset "LA - last layer - full hessian - empfisher" begin
            la = Laplace(
                nn;
                likelihood=:classification,
                hessian_structure=:full,
                subset_of_weights=:last_layer,
                backend=:EmpiricalFisher,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_multi_ll_full_empfisher")
            @test isapprox(
                pytorch_hessian, rearrange_hessian_last_layer(la.H, nn); atol=0.0005
            )
        end

        @testset "LA - last layer - full hessian - ggn" begin
            la = Laplace(
                nn;
                likelihood=:classification,
                hessian_structure=:full,
                subset_of_weights=:last_layer,
                backend=:GGN,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_multi_ll_full_ggn")
            @test isapprox(
                pytorch_hessian, rearrange_hessian_last_layer(la.H, nn); atol=0.0005
            )
        end

        @testset "LA - subnetwork - full hessian - ggn" begin
            la = Laplace(
                nn;
                likelihood=:classification,
                hessian_structure=:full,
                subset_of_weights=:subnetwork,
                subnetwork_indices=[
                    [1, 1, 1],
                    [1, 3, 2],
                    [2, 1],
                    [2, 3],
                    [3, 1, 1],
                    [3, 4, 3],
                    [4, 1],
                    [4, 4],
                ],
                backend=:GGN,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_multi_subnet_full_ggn")
            @test isapprox(pytorch_hessian, la.H; atol=0.0001)
        end

        @testset "LA - subnetwork - full hessian - empfisher" begin
            la = Laplace(
                nn;
                likelihood=:classification,
                hessian_structure=:full,
                subset_of_weights=:subnetwork,
                subnetwork_indices=[
                    [1, 1, 1],
                    [1, 3, 2],
                    [2, 1],
                    [2, 3],
                    [3, 1, 1],
                    [3, 4, 3],
                    [4, 1],
                    [4, 4],
                ],
                backend=:EmpiricalFisher,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_multi_subnet_full_empfisher")
            @test isapprox(pytorch_hessian, la.H; atol=0.0001)
        end
    end

    @testset "Regression" begin

        # Read the dataset from a CSV file
        df = CSV.read(joinpath(@__DIR__, "datafiles", "data_regression.csv"), DataFrame)
        x = df[:, 1]
        y = df[:, 2]

        xs = [[x] for x in x]

        data = zip(xs, y)

        # Read the network weights and biases from a JLB file
        nn = deserialize(joinpath(@__DIR__, "datafiles", "nn-binary_regression.jlb"))

        @testset "LA - full weights - full hessian - ggn" begin
            la = Laplace(
                nn;
                likelihood=:regression,
                hessian_structure=:full,
                subset_of_weights=:all,
                backend=:GGN,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_regression_all_full_ggn")
            @test isapprox(pytorch_hessian, la.H; atol=0.05)
        end

        @testset "LA - last layer - full hessian - ggn" begin
            la = Laplace(
                nn;
                likelihood=:regression,
                hessian_structure=:full,
                subset_of_weights=:last_layer,
                backend=:GGN,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_regression_ll_full_ggn")
            @test isapprox(pytorch_hessian, la.H; atol=0.0005)
        end

        @testset "LA - subnetwork - full hessian - ggn" begin
            la = Laplace(
                nn;
                likelihood=:regression,
                hessian_structure=:full,
                subset_of_weights=:subnetwork,
                subnetwork_indices=[
                    [1, 2, 1], [1, 4, 1], [1, 6, 1], [1, 7, 1], [1, 8, 1], [1, 10, 1]
                ],
                backend=:GGN,
            )
            fit!(la, data)
            pytorch_hessian = read_hessian_csv("hessian_regression_subnet_full_ggn")
            @test isapprox(pytorch_hessian, la.H; atol=0.01)
        end
    end
end
