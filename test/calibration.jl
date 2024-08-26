using Statistics
using LaplaceRedux
using Distributions
using Trapz



# Test for `sharpness_regression` function
@testset "sharpness_regression distributions tests" begin
    @info " testing sharpness_regression with distributions"

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
    @info " testing empirical_frequency_regression with distributions"
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
    @info " testing sharpness_classification with distributions"

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
    @info " testing empirical_frequency_classification with distributions"
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
    #Note to the reader: this is a weak test because it only check if the final average is close to 0.5, but it is a necessary condition.
    #step 1: generate random probabilities
    target_class_probabilities = rand(5000)
    distributions = map(x -> Distributions.Bernoulli(x), target_class_probabilities)
    #step 2: generate perfectly calibrated synthetic output 
    y_binary = []
    for el in distributions
        push!(y_binary, rand(el, 1))
    end
    #step 3: convert to an array of 0 and 1
    y_int = [Int64(first(inner_vector)) for inner_vector in y_binary]
    #step 4: compute empirical average
    num_p_per_interval, emp_avg, bin_centers = empirical_frequency_binary_classification(
        y_int, distributions; n_bins=20
    )

    @test isapprox(mean(emp_avg), 0.5; atol=0.1)

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

# Test for `empirical_frequency_binary_classification` function
@testset "sigma scaling" begin
    @info "testing sigma scaling technique"
    # Test 1: testing function extract_mean_and_variance
    # Create 3 different Normal distributions with known means and variances
    known_distributions = [Normal(0.0, 1.0), Normal(2.0, 3.0), Normal(-1.0, 0.5)]
    expected_means = [0.0, 2.0, -1.0]
    expected_variances = [1.0, 9.0, 0.25]
    # Execution: Call the function
    actual_means, actual_variances = extract_mean_and_variance(known_distributions)
    @test actual_means ≈ expected_means
    @test actual_variances ≈ expected_variances
    # Test 2: testing sigma_scaling 
    # Step 1: Define the parameters for the sine wave
    start_point = 0.0  # Start of the interval
    end_point = 2 * π  # End of the interval, 2π for a full sine wave cycle
    sample_points = 2000  # Number of sample points between 0 and 2π

    # Step 2: Generate the sample points
    x = LinRange(start_point, end_point, sample_points)

    # Step 3: Generate the sine wave data
    y = sin.(x)
    distrs = Distributions.Normal.(y, 0.01)
    #fake  miscalibrated predictions 
    predicted_elements = rand.(distrs) .+ rand((1, 2))

    sigma = sigma_scaling(distrs, predicted_elements)
    @test typeof(sigma) <: Number
end
