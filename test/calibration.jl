using Statistics
using LaplaceRedux

@testset "sharpness_classification tests" begin
    y_binary = [0, 1, 0, 1, 1, 0, 1, 0]
    sampled_distributions = [
            0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8;
            0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2
    ]
    mean_class_one, mean_class_zero = sharpness_classification(y_binary, sampled_distributions)
    @test mean_class_one ≈ mean(sampled_distributions[1,[2,4,5,7]])
    @test mean_class_zero ≈ mean(sampled_distributions[2,[1,3,6,8]])
    
end



# Test for `sharpness_regression` function
@testset "sharpness_regression tests" begin
    sampled_distributions = [[0.1, 0.2, 0.3, 0.7, 0.6], [0.2, 0.3, 0.4, 0.3 , 0.5 ], [0.3, 0.4, 0.5, 0.9, 0.2]]
    mean_variance = mean(map(var, sampled_distributions))
    sharpness = sharpness_regression(sampled_distributions)

    @test sharpness ≈ mean_variance
end