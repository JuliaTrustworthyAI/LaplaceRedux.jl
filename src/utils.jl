using Flux

"""
    get_loss_fun(likelihood::Symbol)

Helper function to choose loss function based on specified model `likelihood`.
"""
function get_loss_fun(likelihood::Symbol, model::Chain)
    if likelihood==:regression
        loss_type = :mse
    else
        if outdim(model) == 1
            loss_type = :logitbinarycrossentropy                        # where the formula for logit binary cross entropy is LBCE = -(y * log(sigmoid(z)) + (1 - y) * log(1 - sigmoid(z)))
        else                                                            # and y is the true binary label (either 0 or 1), z is the logit (unactivated output of last layer) 
            loss_type = :logitcrossentropy                              # where the formula for logit cross entropy is LCE = -sum(yᵢ * log(softmax(z)ᵢ))
        end                                                             # and yᵢ is the true probability of the i-th class, z is the vector of logits
    end

    loss(x, ytrue) = getfield(Flux.Losses, loss_type)(model(x), ytrue)

    return loss
end

""" 
    outdim(model::Chain)

Helper function to determine the output dimension of a `Flux.Chain`,
corresponding to the number of neurons on the last layer of the NN.
"""
function outdim(model::Chain)
    return [size(p) for p in Flux.params(model)][end][1]
end