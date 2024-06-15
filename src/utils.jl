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
cumulative distribution function of the predicted distribution targeting y_t. The function was  suggested by Kuleshov(2018) in https://arxiv.org/abs/1807.00263
    Arguments:
    Y_val: a vector of values y_t
    array: an array of sampled distributions F(x_t) stacked column-wise.
"""
function empirical_frequency(Y_cal,sampled_distributions)


    quantiles= collect(0:0.05:1)
    quantiles_matrix = hcat([quantile(samples, quantiles) for samples in sampled_distributions]...)
    n_rows  = size(bounds_quantiles_matrix,1)
    counts = []

    for i in  1:n_rows
        push!(counts, sum(Y_cal .<= quantiles_matrix[i, :]) / length(Y_cal))
    end
    return counts
end

""" 
    sharpness(array)

FOR REGRESSION MODELS.
Given a calibration dataset (x_t, y_t) for i ∈ {1,...,T} and an array of predicted distributions, the function calculates the 
sharpness of the predicted distributions, i.e., the average of the variances var(F_t) predicted by the forecaster for each x_t
The function was  suggested by Kuleshov(2018) in https://arxiv.org/abs/1807.00263
    Arguments:
    sampled_distributions: an array of sampled distributions F(x_t) stacked column-wise.
"""
function sharpness(sampled_distributions)
    sharpness=  mean(var.(sampled_distributions))
    return sharpness

end




""" 
    empirical_frequency-classification(Y_val,array)

FOR BINARY CLASSIFICATION MODELS.
Given a calibration dataset (x_t, y_t) for i ∈ {1,...,T} let p_t= H(x_t)∈[0,1] be the forecasted probability. 
We group the p_t into intervals I-j for j= 1,2,...,m that form a partition of [0,1]. The function computes
the observed average p_j= T^-1_j ∑_{t:p_t ∈ I_j} y_j in each interval I_j. 
The function was  suggested by Kuleshov(2018) in https://arxiv.org/abs/1807.00263
    Arguments:
    y_binary: the array of outputs y_t numerically coded . 1 for the target class, 0 for the negative result.

    sampled_distributions: an array of sampled distributions stacked column-wise where in the first row 
    there is the probability for the target class y_1=1 and in the second row y_0=0.
"""
function empirical_frequency_binary_classification(y_binary,sampled_distributions)

    pred_avg= collect(range(0,step=0.1,stop=0.9))
    emp_avg = []
    total_pj_per_intervalj = []
    class_probs = sampled_distributions[1, :]

    for j in 1:10
        j_float = j / 10.0 -0.1
        push!(total_pj_per_intervalj,sum( j_float.<class_probs.<j_float+0.1))
       
    
        if total_pj_per_intervalj[j]== 0
            #println("it's zero $j")
            push!(emp_avg, 0)
            #push!(pred_avg, 0)
        else
            indices = findall(x -> j_float < x <j_float+0.1, class_probs)
    
    
    
            push!(emp_avg, 1/total_pj_per_intervalj[j]  *  sum(y_binary[indices]))
            println(" numero $j")
            pred_avg[j] = 1/total_pj_per_intervalj[j]  *  sum(sampled_distributions[1,indices])
        end
    
    end

    return (total_pj_per_intervalj,emp_avg,pred_avg)

end




""" 
    sharpness-classification(array)

FOR BINARY CLASSIFICATION MODELS.
We can also assess sharpness by looking at the distribution of model predictions.When forecasts are sharp, 
most predictions are close to 0 or 1; unsharp forecasters make predictions closer to 0.5.
The function was  suggested by Kuleshov(2018) in https://arxiv.org/abs/1807.00263

    Arguments:
    y_binary: the array of outputs y_t numerically coded . 1 for the target class, 0 for the negative result.
    sampled_distributions: an array of sampled distributions stacked column-wise where in the first row 
    there is the probability for the target class y_1=1 and in the second row y_0=0.
"""
function sharpness_classification(y_binary,sampled_distributions)

    class_one = sampled_distributions[1,findall(y_binary .== 1)]
    class_zero = sampled_distributions[2,findall(y_binary .== 0)]
    
    return mean(class_one),mean(class_zero)
    
end