using Flux
using Optimisers
using Statistics
using Zygote: gradient, jacobian

"""
    collect_trainable(model)

Collect all trainable parameter arrays from a Flux model, in the same order
as `Flux.destructure`. This replaces the deprecated `Flux.params` API.
"""
function collect_trainable(model)
    ps = AbstractArray[]
    _collect_trainable!(ps, Flux.trainable(model))
    return ps
end

function _collect_trainable!(ps, x::AbstractArray{<:AbstractFloat})
    push!(ps, x)
end

# Non-float arrays (e.g., Int, Bool) are not trainable parameters
_collect_trainable!(ps, ::AbstractArray) = nothing

function _collect_trainable!(ps, x::NamedTuple)
    for v in values(x)
        _collect_trainable!(ps, v)
    end
end

function _collect_trainable!(ps, x::Tuple)
    for v in x
        _collect_trainable!(ps, v)
    end
end

# Skip non-parametric leaves
_collect_trainable!(ps, ::Function) = nothing
_collect_trainable!(ps, ::Nothing) = nothing
_collect_trainable!(ps, ::Number) = nothing

# For Flux layers and other objects, recurse via Flux.trainable
function _collect_trainable!(ps, x)
    t = Flux.trainable(x)
    if t isa NamedTuple || t isa Tuple
        _collect_trainable!(ps, t)
    end
end

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
    # Walk backwards through layers to find the last parametric layer
    for layer in reverse(model.layers)
        if hasproperty(layer, :weight)
            return size(layer.weight, 1)
        end
    end
    error("Could not determine output dimension: no parametric layers found")
end
