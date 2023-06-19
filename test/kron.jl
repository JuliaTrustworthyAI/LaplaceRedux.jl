using LaplaceRedux.Curvature: Kron

@testset "Addition one block" begin
    left = Kron([([1 2], [3 4])])
    right = Kron([([5 6], [7 8])])
    total = left + right
    @test total.kfacs == [([6 8], [10 12])]
end

@testset "Addition two blocks" begin
    left = Kron([([1 2], [3 4]), ([11 12], [13 14])])
    right = Kron([([5 6], [7 8]), ([15 16], [17 18])])
    total = left + right
    @test total.kfacs == [([6 8], [10 12]), ([26 28], [30 32])]
end

@testset "Addition empty" begin
    left = Kron([])
    total = left + left
    @test total == left
end

@testset "Addition invalid" begin
    left = Kron([([1 2], [3 4]), ([], [])])
    right = Kron([([5 6], [7 8])])
    @test_throws AssertionError left + right
end

@testset "Scaling" begin
    left = 4
    right = Kron([([5 6], [7 8])])
    @test left * right == right * left
    @test (left * right).kfacs == [([10 12], [14 16])]
end
