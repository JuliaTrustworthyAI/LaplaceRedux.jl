using LaplaceRedux.Curvature: Kron, KronDecomposed, decompose, detblock, det
using LinearAlgebra

@testset "Decomposition, scaling" begin
    K = Kron([([-1 0; 0 2], [3 0; 0 4])])
    # Eigenvalues are (1,2) and (3,4), as on the diags.
    # Eigenvectors are the basis vectors
    KD = decompose(K)
    # The negative eigenvalue clamped
    @test KD.delta == 0
    @test KD[1][1].values ≈ [0, 2]
    @test KD[1][2].values ≈ [3, 4]
    @test KD[1][1].vectors ≈ KD[1][2].vectors ≈ [1 0; 0 1]
    @test length(KD) == 1
    KD4 = KD * 4
    @test KD4[1][1].values ≈ (4KD)[1][1].values
    @test KD4.delta == 0
end

@testset "Addition" begin
    K = Kron([([-1 0; 0 2], [3 0; 0 4])])
    KD = decompose(K)
    KD_plus = KD + Diagonal([9, 9])
    @test KD_plus.delta == 9
    @test 9 + KD == KD + 9
end

@testset "Determinant" begin
    block = ([-1 0; 0 2], [3 0; 0 4])
    K = Kron([block, 2 .* block])
    delta = d = 3
    KD = decompose(K) + delta
    det_1 = detblock(KD[1], delta)
    det_2 = detblock(KD[2], delta)
    @test det_1 == 2 * (3 + 4) + 4d
    @test det_2 == (2^2) * 2 * (3 + 4) + 4d
    @test det(KD) == det_1 + det_2
end
