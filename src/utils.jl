using Flux
using Statistics

"""
    get_loss_fun(likelihood::Symbol)

Helper function to choose loss function based on specified model `likelihood`.
"""
function get_loss_fun(likelihood::Symbol, model::Chain)::Function
    loss_type = get_loss_type(likelihood, model)
    flux_loss = getfield(Flux.Losses, loss_type)

    return flux_loss
end

"""
    get_loss_type(likelihood::Symbol)

Choose loss function type based on specified model `likelihood`.
"""
function get_loss_type(likelihood::Symbol, model::Chain)::Symbol
    if likelihood == :regression
        loss_type = :mse
    else
        if outdim(model) == 1
            loss_type = :logitbinarycrossentropy                        # where the formula for logit binary cross entropy is LBCE = -(y * log(sigmoid(z)) + (1 - y) * log(1 - sigmoid(z)))
        else                                                            # and y is the true binary label (either 0 or 1), z is the logit (unactivated output of last layer) 
            loss_type = :logitcrossentropy                              # where the formula for logit cross entropy is LCE = -sum(yᵢ * log(softmax(z)ᵢ))
        end                                                             # and yᵢ is the true probability of the i-th class, z is the vector of logits
    end
    return loss_type
end

""" 
    outdim(model::Chain)

Helper function to determine the output dimension of a `Flux.Chain`,
corresponding to the number of neurons on the last layer of the NN.
"""
function outdim(model::Chain)::Number
    return [size(p) for p in Flux.params(model)][end][1]
end

""" 
    empirical_frequency(Y_val,array)

FOR REGRESSION MODELS.
Given a calibration dataset (x_t, y_t) for i ∈ {1,...,T} and an array of predicted distributions, the function calculates the empirical frequency
phat_j = {y_t|F_t(y_t)<= p_j, t= 1,....,T}/T, where T is the number of calibration points, p_j is the confidence level and F_t is the 
cumulative density function of the predicted distribution targeting y_t. The function was  suggested by Kuleshov(2018) in https://arxiv.org/abs/1807.00263
    Arguments:
    Y_val: a vector of values
    array: an array of sampled distributions stacked column-wise.
"""
function empirical_frequency(Y_cal,array)


    quantiles= collect(0:0.05:1)
    quantiles_matrix = hcat([quantile(samples, quantiles) for samples in array]...)
    n_rows  = size(bounds_quantiles_matrix,1)
    counts = []

    for i in  1:n_rows
        push!(counts, sum(Y_cal .<= quantiles_matrix[i, :]) / length(Y_cal))
    end
    return counts
end