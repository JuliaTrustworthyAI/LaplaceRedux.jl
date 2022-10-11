using LaplaceRedux.Data

fun_list = [
    Data.toy_data_linear,
    Data.toy_data_non_linear,
    Data.toy_data_multi,
]

for fun in fun_list
    N = 100
    xs, ys = fun()
    @test isa(xs, Vector)
    @test isa(ys, AbstractArray)
    @test size(xs, 1) == N
    @test length(ys) == N 
end