using Statistics
using LaplaceRedux
using Distributions

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



# Test for `empirical_frequency_regression` function
@testset "empirical_frequency_regression tests" begin
    # Test 1: Check that the function runs without errors and returns an array for a simple case
    Y_cal = [0.5, 1.5, 2.5, 3.5, 4.5]
    n_bins = 10
    sampled_distributions = [rand(Distributions.Normal(1, 1.0),6) for _ in 1:5]
    counts = empirical_frequency_regression(Y_cal, sampled_distributions, n_bins=n_bins)
    @test typeof(counts) == Array{Float64, 1}  # Check if the output is an array of Float64
    @test length(counts) == n_bins + 1 

    # Test 2: Check the function with a known input
    #to do

    # Test 3: Invalid n_bins input
    Y_cal = [0.5, 1.5, 2.5, 3.5, 4.5]
    sampled_distributions =  [rand(Distributions.Normal(1, 1.0),6) for _ in 1:5]
    @test_throws ArgumentError empirical_frequency_regression(Y_cal, sampled_distributions, n_bins=0)

end



# Test for `empirical_frequency_binary_classification` function
@testset "empirical_frequency_binary_classification tests" begin
    # Test 1: Check that the function runs without errors and returns an array for a simple case
    y_binary = rand(0:1, 10)
    sampled_distributions = rand(2,10)
    n_bins = 4
    num_p_per_interval, emp_avg, bin_centers = empirical_frequency_binary_classification(y_binary, sampled_distributions,  n_bins=n_bins)
    @test length(num_p_per_interval) == n_bins
    @test length(emp_avg) == n_bins
    @test length(bin_centers) == n_bins

    # Test 2: Check the function with a known input

    #to do



    # Test 3: Invalid Y_cal input
    Y_cal =  [0, 1, 0, 1.2, 4]
    sampled_distributions =  rand(2,5)
    @test_throws ArgumentError empirical_frequency_binary_classification(Y_cal, sampled_distributions, n_bins=10)


    # Test 4: Invalid n_bins input
    Y_cal = rand(0:1, 5)
    sampled_distributions =  rand(2,5)
    @test_throws ArgumentError empirical_frequency_binary_classification(Y_cal, sampled_distributions, n_bins=0)
end