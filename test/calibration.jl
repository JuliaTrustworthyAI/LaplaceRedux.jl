using Statistics
using LaplaceRedux
using Distributions
using Trapz

@testset "sharpness_classification sampled distributions tests" begin

    # Test 1: Check that the function runs without errors and returns two scalars for a simple case
    y_binary = [1, 0, 1, 0, 1]
    sampled_distributions = [0.9 0.1 0.8 0.2 0.7; 0.1 0.9 0.2 0.8 0.3]  # Sampled probabilities
    mean_class_one, mean_class_zero = sharpness_classification(
        y_binary, sampled_distributions
    )
    @test typeof(mean_class_one) <: Real  # Check if mean_class_one is a scalar
    @test typeof(mean_class_zero) <: Real  # Check if mean_class_zero is a scalar

    # Test 2: Check the function with a known input
    y_binary = [0, 1, 0, 1, 1, 0, 1, 0]
    sampled_distributions = [
        0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
        0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2
    ]
    mean_class_one, mean_class_zero = sharpness_classification(
        y_binary, sampled_distributions
    )
    @test mean_class_one ≈ mean(sampled_distributions[1, [2, 4, 5, 7]])
    @test mean_class_zero ≈ mean(sampled_distributions[2, [1, 3, 6, 8]])

    # Test 3: Edge case with all ones in y_binary
    y_binary_all_ones = [1, 1, 1]
    sampled_distributions_all_ones = [0.8 0.9 0.7; 0.2 0.1 0.3]
    mean_class_one_all_ones, mean_class_zero_all_ones = sharpness_classification(
        y_binary_all_ones, sampled_distributions_all_ones
    )
    @test mean_class_one_all_ones == mean([0.8, 0.9, 0.7])
    @test isnan(mean_class_zero_all_ones)  # Since there are no zeros in y_binary, the mean should be NaN

    # Test 4: Edge case with all zeros in y_binary
    y_binary_all_zeros = [0, 0, 0]
    sampled_distributions_all_zeros = [0.1 0.2 0.3; 0.9 0.8 0.7]
    mean_class_one_all_zeros, mean_class_zero_all_zeros = sharpness_classification(
        y_binary_all_zeros, sampled_distributions_all_zeros
    )
    @test mean_class_zero_all_zeros == mean([0.9, 0.8, 0.7])
    @test isnan(mean_class_one_all_zeros)  # Since there are no ones in y_binary, the mean should be NaN
end

# Test for `sharpness_regression` function
@testset "sharpness_regression sampled distributions tests" begin

    # Test 1: Check that the function runs without errors and returns a scalar for a simple case
    sampled_distributions = [randn(100) for _ in 1:10]  # Create 10 distributions, each with 100 samples
    sharpness = sharpness_regression(sampled_distributions)
    @test typeof(sharpness) <: Real  # Check if the output is a scalar

    # Test 2: Check the function with a known input
    sampled_distributions = [
        [0.1, 0.2, 0.3, 0.7, 0.6], [0.2, 0.3, 0.4, 0.3, 0.5], [0.3, 0.4, 0.5, 0.9, 0.2]
    ]
    mean_variance = mean(map(var, sampled_distributions))
    sharpness = sharpness_regression(sampled_distributions)
    @test sharpness ≈ mean_variance

    # Test 3: Edge case with identical distributions
    sampled_distributions_identical = [ones(100) for _ in 1:10]  # Identical distributions, zero variance
    sharpness_identical = sharpness_regression(sampled_distributions_identical)
    @test sharpness_identical == 0.0  # Sharpness should be zero for identical distributions
end

# Test for `empirical_frequency_regression` function
@testset "empirical_frequency_regression sampled distributions tests" begin
    # Test 1: Check that the function runs without errors and returns an array for a simple case
    Y_cal = [0.5, 1.5, 2.5, 3.5, 4.5]
    n_bins = 10
    sampled_distributions = [rand(Distributions.Normal(1, 1.0), 6) for _ in 1:5]
    counts = empirical_frequency_regression(Y_cal, sampled_distributions; n_bins=n_bins)
    @test typeof(counts) == Array{Float64,1}  # Check if the output is an array of Float64
    @test length(counts) == n_bins + 1

    # Test 2: Check the function with a known input
    #to do
    # Step 1: Define the parameters for the sine wave
    start_point = 0.0  # Start of the interval
    end_point = 2 * π  # End of the interval, 2π for a full sine wave cycle
    sample_points = 2000  # Number of sample points between 0 and 2π

    # Step 2: Generate the sample points
    x = LinRange(start_point, end_point, sample_points)

    # Step 3: Generate the sine wave data
    y = sin.(x)
    # Step 4: Generate samples
    distrs = Distributions.Normal.(y, 0.01)
    sampled_distributions = rand.(distrs, 100)
    #fake perfectly calibrated predictions 
    predicted_elements = rand.(distrs)
    n_bins = 100
    emp_freq = empirical_frequency_regression(
        predicted_elements, sampled_distributions; n_bins=n_bins
    )
    quantiles = collect(range(0; stop=1, length=n_bins + 1))
    area = trapz((quantiles), vec(abs.(emp_freq - quantiles)))
    @test area < 0.1

    # Test 3: Invalid n_bins input
    Y_cal = [0.5, 1.5, 2.5, 3.5, 4.5]
    sampled_distributions = [rand(Distributions.Normal(1, 1.0), 6) for _ in 1:5]
    @test_throws ArgumentError empirical_frequency_regression(
        Y_cal, sampled_distributions, n_bins=0
    )
end

# Test for `empirical_frequency_binary_classification` function
@testset "empirical_frequency_binary_classification sampled distributions tests" begin
    # Test 1: Check that the function runs without errors and returns an array for a simple case
    y_binary = rand(0:1, 10)
    sampled_distributions = rand(2, 10)
    n_bins = 4
    num_p_per_interval, emp_avg, bin_centers = empirical_frequency_binary_classification(
        y_binary, sampled_distributions; n_bins=n_bins
    )
    @test length(num_p_per_interval) == n_bins
    @test length(emp_avg) == n_bins
    @test length(bin_centers) == n_bins

    # Test 2: Check the function with a known input

    #to do

    # Test 3: Invalid Y_cal input
    Y_cal = [0, 1, 0, 1.2, 4]
    sampled_distributions = rand(2, 5)
    @test_throws ArgumentError empirical_frequency_binary_classification(
        Y_cal, sampled_distributions, n_bins=10
    )

    # Test 4: Invalid n_bins input
    Y_cal = rand(0:1, 5)
    sampled_distributions = rand(2, 5)
    @test_throws ArgumentError empirical_frequency_binary_classification(
        Y_cal, sampled_distributions, n_bins=0
    )
end

# Test for `sharpness_regression` function
@testset "sharpness_regression distributions tests" begin

    # Test 1: Check that the function runs without errors and returns a scalar for a simple case
    distributions = [Distributions.Normal.(1, 0.01) for _ in 1:10]  # Create 10 distributions, each with 100 samples
    sharpness = sharpness_regression(distributions)
    @test typeof(sharpness) <: Real  # Check if the output is a scalar

    # Test 2: Check the function with a known input
    distributions = [
        Distributions.Normal.(1, 0.01),
        Distributions.Normal.(1, 2),
        Distributions.Normal.(1, 3),
    ]
    mean_variance = mean(map(var, distributions))
    sharpness = sharpness_regression(distributions)
    @test sharpness ≈ mean_variance
end
# Test for `empirical_frequency_regression` function
@testset "empirical_frequency_regression distributions tests" begin
    # Test 1: Check that the function runs without errors and returns an array for a simple case
    Y_cal = [0.5, 1.5, 2.5, 3.5, 4.5]
    n_bins = 10
    distributions = [Distributions.Normal(1, 1.0) for _ in 1:5]
    counts = empirical_frequency_regression(Y_cal, distributions; n_bins=n_bins)
    @test typeof(counts) == Array{Float64,1}  # Check if the output is an array of Float64
    @test length(counts) == n_bins + 1

    # Test 2: Check the function with a known input

    # Step 1: Define the parameters for the sine wave
    start_point = 0.0  # Start of the interval
    end_point = 2 * π  # End of the interval, 2π for a full sine wave cycle
    sample_points = 2000  # Number of sample points between 0 and 2π

    # Step 2: Generate the sample points
    x = LinRange(start_point, end_point, sample_points)

    # Step 3: Generate the sine wave data
    y = sin.(x)
    distrs = Distributions.Normal.(y, 0.01)
    #fake perfectly calibrated predictions 
    predicted_elements = rand.(distrs)
    n_bins = 100
    emp_freq = empirical_frequency_regression(predicted_elements, distrs; n_bins=n_bins)
    quantiles = collect(range(0; stop=1, length=n_bins + 1))
    area = trapz((quantiles), vec(abs.(emp_freq - quantiles)))
    @test area < 0.1

    # Test 3: Invalid n_bins input
    Y_cal = [0.5, 1.5, 2.5, 3.5, 4.5]
    distributions = [Distributions.Normal(1, 1.0) for _ in 1:5]
    @test_throws ArgumentError empirical_frequency_regression(
        Y_cal, distributions, n_bins=0
    )
end
@testset "sharpness_classification distributions tests" begin

    # Test 1: Check that the function runs without errors and returns two scalars for a simple case
    y_binary = [1, 0, 1, 0]
    distributions = [
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.7)
    ]  # probabilities
    mean_class_one, mean_class_zero = sharpness_classification(y_binary, distributions)
    @test typeof(mean_class_one) <: Real  # Check if mean_class_one is a scalar
    @test typeof(mean_class_zero) <: Real  # Check if mean_class_zero is a scalar

    # Test 2: Check the function with a known input
    y_binary = [0, 1, 0, 1, 1]
    distributions = [
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.3)
        Distributions.Bernoulli(0.5)
        Distributions.Bernoulli(0.9)
    ]
    mean_class_one, mean_class_zero = sharpness_classification(y_binary, distributions)
    @test mean_class_one ≈ mean(mean.(distributions[[2, 4, 5]]))
    @test mean_class_zero ≈ mean(mean.(distributions[[1, 3]]))

    # Test 3: Edge case with all ones in y_binary
    y_binary_all_ones = [1, 1, 1]
    distributions_all_ones = [
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.3)
    ]
    mean_class_one_all_ones, mean_class_zero_all_ones = sharpness_classification(
        y_binary_all_ones, distributions_all_ones
    )
    @test mean_class_one_all_ones ≈ mean([0.7, 0.7, 0.3])
    @test isnan(mean_class_zero_all_ones)  # Since there are no zeros in y_binary, the mean should be NaN

    # Test 4: Edge case with all zeros in y_binary
    y_binary_all_zeros = [0, 0, 0]
    distributions_all_zeros = [
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.7)
        Distributions.Bernoulli(0.3)
    ]
    mean_class_one_all_zeros, mean_class_zero_all_zeros = sharpness_classification(
        y_binary_all_zeros, distributions_all_zeros
    )
    @test mean_class_zero_all_zeros ≈ mean([0.3, 0.3, 0.7])
    @test isnan(mean_class_one_all_zeros)  # Since there are no ones in y_binary, the mean should be NaN
end

# Test for `empirical_frequency_binary_classification` function
@testset "empirical_frequency_binary_classification distributions tests" begin
    # Test 1: Check that the function runs without errors and returns an array for a simple case
    y_binary = rand(0:1, 10)
    distributions = [Distributions.Bernoulli(0.7) for _ in 1:10]
    n_bins = 4
    num_p_per_interval, emp_avg, bin_centers = empirical_frequency_binary_classification(
        y_binary, distributions; n_bins=n_bins
    )
    @test length(num_p_per_interval) == n_bins
    @test length(emp_avg) == n_bins
    @test length(bin_centers) == n_bins

    # Test 2: Check the function with a known input

    #to do

    # Test 3: Invalid Y_cal input
    Y_cal = [0, 1, 0, 1.2, 4]
    distributions = [Distributions.Bernoulli(0.7) for _ in 1:5]
    @test_throws ArgumentError empirical_frequency_binary_classification(
        Y_cal, distributions, n_bins=10
    )

    # Test 4: Invalid n_bins input
    Y_cal = rand(0:1, 5)
    distributions = [Distributions.Bernoulli(0.7) for _ in 1:5]
    @test_throws ArgumentError empirical_frequency_binary_classification(
        Y_cal, distributions, n_bins=0
    )
end
