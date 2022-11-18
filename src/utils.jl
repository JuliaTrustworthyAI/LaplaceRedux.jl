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
            loss_type = :logitbinarycrossentropy
        else
            loss_type = :logitcrossentropy 
        end
    end

    loss(x, ytrue) = getfield(Flux.Losses, loss_type)(model(x), ytrue)

    return loss
end

"""
    outdim(model::Chain)

Helper function to determine the output dimension of a `Flux.Chain`
"""
function outdim(model::Chain)
    return [size(p) for p in Flux.params(model)][end][1]
end