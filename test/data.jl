using LaplaceRedux.Data
using Random

fun_list = [Data.toy_data_linear, Data.toy_data_non_linear, Data.toy_data_multi]

for fun in fun_list
    N = 100
    xs, ys = fun()
    @test isa(xs, Vector)
    @test isa(ys, AbstractArray)
    @test size(xs, 1) == N
    @test length(ys) == N
    seed = 1234

    # Generate data with the same seed
    Random.seed!(seed)
    xs1, ys1 = fun(N; seed=seed)
    
    Random.seed!(seed)
    xs2, ys2 = fun(N; seed=seed)

    # Test that the outputs are the same
    @test xs1 == xs2
    @test ys1 == ys2
end
