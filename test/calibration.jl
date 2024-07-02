using Statistics
using LaplaceRedux

@testset "sharpness_classification tests" begin


    # Test 1: Check that the function runs without errors and returns two scalars for a simple case
    y_binary = [1, 0, 1, 0, 1]
    sampled_distributions = [0.9 0.1 0.8 0.2 0.7; 0.1 0.9 0.2 0.8 0.3]  # Sampled probabilities
    mean_class_one, mean_class_zero = sharpness_classification(y_binary, sampled_distributions)
    @test typeof(mean_class_one) <: Real  # Check if mean_class_one is a scalar
    @test typeof(mean_class_zero) <: Real  # Check if mean_class_zero is a scalar


    # Test 2: Check the function with a known input
    y_binary = [0, 1, 0, 1, 1, 0, 1, 0]
    sampled_distributions = [
            0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8;
            0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2
    ]
    mean_class_one, mean_class_zero = sharpness_classification(y_binary, sampled_distributions)
    @test mean_class_one ≈ mean(sampled_distributions[1,[2,4,5,7]])
    @test mean_class_zero ≈ mean(sampled_distributions[2,[1,3,6,8]])

    # Test 3: Edge case with all ones in y_binary
    y_binary_all_ones = [1, 1, 1]
    sampled_distributions_all_ones = [0.8 0.9 0.7; 0.2 0.1 0.3]
    mean_class_one_all_ones, mean_class_zero_all_ones = sharpness_classification(y_binary_all_ones, sampled_distributions_all_ones)
    @test mean_class_one_all_ones == mean([0.8, 0.9, 0.7])
    @test isnan(mean_class_zero_all_ones)  # Since there are no zeros in y_binary, the mean should be NaN

    # Test 4: Edge case with all zeros in y_binary
    y_binary_all_zeros = [0, 0, 0]
    sampled_distributions_all_zeros = [0.1 0.2 0.3; 0.9 0.8 0.7]
    mean_class_one_all_zeros, mean_class_zero_all_zeros = sharpness_classification(y_binary_all_zeros, sampled_distributions_all_zeros)
    @test mean_class_zero_all_zeros == mean([0.9, 0.8, 0.7])
    @test isnan(mean_class_one_all_zeros)  # Since there are no ones in y_binary, the mean should be NaN

    
end



# Test for `sharpness_regression` function
@testset "sharpness_regression tests" begin

    # Test 1: Check that the function runs without errors and returns a scalar for a simple case
    sampled_distributions = [randn(100) for _ in 1:10]  # Create 10 distributions, each with 100 samples
    sharpness = sharpness_regression(sampled_distributions)
    @test typeof(sharpness) <: Real  # Check if the output is a scalar

    # Test 2: Check the function with a known input
    sampled_distributions = [[0.1, 0.2, 0.3, 0.7, 0.6], [0.2, 0.3, 0.4, 0.3 , 0.5 ], [0.3, 0.4, 0.5, 0.9, 0.2]]
    mean_variance = mean(map(var, sampled_distributions))
    sharpness = sharpness_regression(sampled_distributions)
    @test sharpness ≈ mean_variance

    # Test 3: Edge case with identical distributions
    sampled_distributions_identical = [ones(100) for _ in 1:10]  # Identical distributions, zero variance
    sharpness_identical = sharpness_regression(sampled_distributions_identical)
    @test sharpness_identical == 0.0  # Sharpness should be zero for identical distributions
    

end