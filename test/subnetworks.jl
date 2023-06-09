using LaplaceRedux
using LaplaceRedux.Curvature
using LaplaceRedux.Data
using Flux

@testset "Subnetwork Laplace" begin

    @testset "Incorrect Indices Errors" begin
        nn = Chain(Dense(2, 2, σ), Dense(2, 1))
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
    end

    @testset "Testing index conversion on all indices" begin
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
        @test la.curvature.subnetwork_indices == collect(1:41)
    end
end