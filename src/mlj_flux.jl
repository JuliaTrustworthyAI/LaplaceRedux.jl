using Flux
using MLJFlux
import MLJModelInterface as MMI
using ProgressMeter: Progress, next!, BarGlyphs
using Random
using Tables
using ComputationalResources
using Statistics

mutable struct LaplaceApproximation{B,F,O,L} <: MLJFlux.MLJFluxProbabilistic
    builder::B
    finaliser::F
    optimiser::O   # mutable struct from Flux/src/optimise/optimisers.jl
    loss::L        # can be called as in `loss(yhat, y)`
    epochs::Int    # number of epochs
    batch_size::Int  # size of a batch
    lambda::Float64  # regularization strength
    alpha::Float64   # regularizaton mix (0 for all l2, 1 for all l1)
    rng::Union{AbstractRNG,Int64}
    optimiser_changes_trigger_retraining::Bool
    acceleration::AbstractResource  # eg, `CPU1()` or `CUDALibs()`
    likelihood::Symbol
    subset_of_weights::Symbol
    subnetwork_indices::Vector{Vector{Int}}
    hessian_structure::Symbol
    backend::Symbol
    σ::Real
    μ₀::Real
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}
    link_approx::Symbol
    fit_params::Dict{Symbol,Any}
    la::Union{Nothing,BaseLaplace}
end

"""
LaplaceApproximation(;
builder::B,
finaliser::F,
optimiser::O,
loss::L,
epochs::Int,
batch_size::Int,
lambda::Float64,
alpha::Float64,
rng::Union{AbstractRNG,Int64},
optimiser_changes_trigger_retraining::Bool,
acceleration::AbstractResource,
likelihood::Symbol,
subset_of_weights::Symbol,
subnetwork_indices::Vector{Vector{Int}},
hessian_structure::Symbol,
backend::Symbol,
σ::Float64,
μ₀::Float64,
P₀::Union{AbstractMatrix,UniformScaling},
link_approx::Symbol,
fit_params::Dict{Symbol,Any},
) where {B,F,O,L}

Constructor for LaplaceApproximation, a wrapper for Laplace, a bayesian deep learning model.
"""
function LaplaceApproximation(;
    builder::B=MLJFlux.MLP(; hidden=(32, 32, 32), σ=Flux.swish),
    finaliser::F=Flux.softmax,
    optimiser::O=Flux.Optimise.Adam(),
    loss::L=Flux.crossentropy,
    epochs::Int=10,
    batch_size::Int=1,
    lambda::Float64=1.0,
    alpha::Float64=0.0,
    rng::Union{AbstractRNG,Int64}=Random.GLOBAL_RNG,
    optimiser_changes_trigger_retraining::Bool=false,
    acceleration::AbstractResource=CPU1(),
    likelihood::Symbol=:classification,
    subset_of_weights::Symbol=:all,
    subnetwork_indices::Vector{Vector{Int}}=Vector{Vector{Int}}([]),
    hessian_structure::Symbol=:full,
    backend::Symbol=:GGN,
    σ::Float64=1.0,
    μ₀::Float64=0.0,
    P₀::Union{AbstractMatrix,UniformScaling,Nothing}=nothing,
    link_approx::Symbol=:probit,
    fit_params::Dict{Symbol,Any}=Dict{Symbol,Any}(:override => true),
) where {B,F,O,L}
    model = LaplaceApproximation(
        builder,
        finaliser,
        optimiser,
        loss,
        epochs,
        batch_size,
        lambda,
        alpha,
        rng,
        optimiser_changes_trigger_retraining,
        acceleration,
        likelihood,
        subset_of_weights,
        subnetwork_indices,
        hessian_structure,
        backend,
        σ,
        μ₀,
        P₀,
        link_approx,
        fit_params,
        nothing,
    )

    message = MMI.clean!(model)
    isempty(message) || @warn message

    return model
end

function MLJFlux.shape(model::LaplaceApproximation, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    levels = MMI.classes(y[1])
    n_output = length(levels)
    n_input = length(Tables.schema(X).names)
    return (n_input, n_output)
end

function MLJFlux.build(model::LaplaceApproximation, rng, shape)
    # Construct the chain
    chain = Flux.Chain(MLJFlux.build(model.builder, rng, shape...), model.finaliser)
    # Construct Laplace model and store it in the model object
    model.la = Laplace(
        chain;
        likelihood=model.likelihood,
        subset_of_weights=model.subset_of_weights,
        subnetwork_indices=model.subnetwork_indices,
        hessian_structure=model.hessian_structure,
        backend=model.backend,
        σ=model.σ,
        μ₀=model.μ₀,
        P₀=model.P₀,
    )
    return chain
end

function MLJFlux.fitresult(model::LaplaceApproximation, chain, y)
    return (chain, model.la, MMI.classes(y[1]))
end

function MLJFlux.train!(model::LaplaceApproximation, penalty, chain, optimiser, X, y)
    loss = model.loss
    n_batches = length(y)
    training_loss = zero(Float32)
    for i in 1:n_batches
        parameters = Flux.params(chain)
        gs = Flux.gradient(parameters) do
            yhat = chain(X[i])
            batch_loss = loss(yhat, y[i]) + penalty(parameters) / n_batches
            training_loss += batch_loss
            return batch_loss
        end
        Flux.update!(optimiser, parameters, gs)
    end
    return training_loss / n_batches
end

function MLJFlux.fit!(
    model::LaplaceApproximation, penalty, chain, optimiser, epochs, verbosity, X, y
)
    loss = model.loss

    # intitialize and start progress meter:
    meter = Progress(
        epochs + 1;
        dt=0,
        desc="Optimising neural net:",
        barglyphs=BarGlyphs("[=> ]"),
        barlen=25,
        color=:yellow,
    )
    verbosity != 1 || next!(meter)

    # initiate history:
    n_batches = length(y)

    parameters = Flux.params(chain)

    # initial loss:
    losses = (loss(chain(X[i]), y[i]) + penalty(parameters) / n_batches for i in 1:n_batches)
    history = [mean(losses)]

    for i in 1:epochs
        current_loss = MLJFlux.train!(
            model::MLJFlux.MLJFluxModel, penalty, chain, optimiser, X, y
        )
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    la = model.la

    # fit the Laplace model:
    fit!(la, zip(X, y); model.fit_params...)
    optimize_prior!(la; verbose=false, n_steps=100)

    model.la = la

    return chain, history
end

function MMI.clean!(model::LaplaceApproximation)
    warning = ""
    if model.lambda < 0
        warning *= "Need `lambda ≥ 0`. Resetting `lambda = 0`. "
        model.lambda = 0
    end
    if model.alpha < 0 || model.alpha > 1
        warning *= "Need alpha in the interval `[0, 1]`. " * "Resetting `alpha = 0`. "
        model.alpha = 0
    end
    if model.epochs < 0
        warning *= "Need `epochs ≥ 0`. Resetting `epochs = 10`. "
        model.epochs = 10
    end
    if model.batch_size <= 0
        warning *= "Need `batch_size > 0`. Resetting `batch_size = 1`. "
        model.batch_size = 1
    end
    if model.acceleration isa CUDALibs && gpu_isdead()
        warning *=
            "`acceleration isa CUDALibs` " * "but no CUDA device (GPU) currently live. "
    end
    if !(model.acceleration isa CUDALibs || model.acceleration isa CPU1)
        warning *= "`Undefined acceleration, falling back to CPU`"
        model.acceleration = CPU1()
    end
    if model.likelihood ∉ (:classification, :regression)
        warning *=
            "Need `likelihood ∈ (:classification, :regression)`. " *
            "Resetting `likelihood = :classification`. "
        model.likelihood = :classification
    end
    if model.subset_of_weights ∉ (:all, :last_layer, :subnetwork)
        warning *=
            "Need `subset_of_weights ∈ (:all, :last_layer, :subnetwork)`. " *
            "Resetting `subset_of_weights = :all`. "
        model.subset_of_weights = :all
    end
    if model.hessian_structure ∉ (:full, :diagonal)
        warning *=
            "Need `hessian_structure ∈ (:full, :diagonal)`. " *
            "Resetting `hessian_structure = :full`. "
        model.hessian_structure = :full
    end
    if model.backend ∉ (:GGN, :EmpiricalFisher)
        warning *=
            "Need `backend ∈ (:GGN, :EmpiricalFisher)`. " * "Resetting `backend = :GGN`. "
        model.backend = :GGN
    end
    if model.link_approx ∉ (:probit, :plugin)
        warning *=
            "Need `link_approx ∈ (:probit, :plugin)`. " *
            "Resetting `link_approx = :probit`. "
        model.link_approx = :probit
    end
    return warning
end

function MMI.predict(model::LaplaceApproximation, fitresult, Xnew)
    chain, la, levels = fitresult
    # re-format Xnew into acceptable input for Laplace:
    X = MLJFlux.reformat(Xnew)
    # predict using Laplace:
    yhat = vcat(
        [
            predict(la, MLJFlux.tomat(X[:, i]); link_approx=model.link_approx)' for
            i in 1:size(X, 2)
        ]...,
    )
    if la.likelihood == :classification
        # return a UnivariateFinite:
        return MMI.UnivariateFinite(levels, yhat)
    end
    if la.likelihood == :regression
        # return a UnivariateNormal:
        return MMI.UnivariateNormal(yhat[1], sqrt(yhat[2]))
    end
end

function _isdefined(object, name)
    pnames = propertynames(object)
    fnames = fieldnames(typeof(object))
    name in pnames && !(name in fnames) && return true
    return isdefined(object, name)
end

function _equal_to_depth_one(x1, x2)
    names = propertynames(x1)
    names === propertynames(x2) || return false
    for name in names
        getproperty(x1, name) == getproperty(x2, name) || return false
    end
    return true
end

function MMI.is_same_except(
    m1::M1, m2::M2, exceptions::Symbol...
) where {M1<:LaplaceApproximation,M2<:LaplaceApproximation}
    typeof(m1) === typeof(m2) || return false
    names = propertynames(m1)
    propertynames(m2) === names || return false

    for name in names
        if !(name in exceptions) && name != :la
            if !_isdefined(m1, name)
                !_isdefined(m2, name) || return false
            elseif _isdefined(m2, name)
                if name in MLJFlux.deep_properties(M1)
                    _equal_to_depth_one(getproperty(m1, name), getproperty(m2, name)) ||
                        return false
                else
                    (
                        MMI.is_same_except(getproperty(m1, name), getproperty(m2, name)) ||
                        getproperty(m1, name) isa AbstractRNG ||
                        getproperty(m2, name) isa AbstractRNG
                    ) || return false
                end
            else
                return false
            end
        end
    end
    return true
end

MMI.metadata_model(
    LaplaceApproximation;
    input=Union{
        AbstractMatrix{MMI.Continuous},
        MMI.Table(MMI.Continuous),
        MMI.Table{AbstractVector{MMI.Continuous}},
    },
    target=Union{
        AbstractArray{MMI.Finite},
        AbstractArray{MMI.Continuous},
        AbstractVector{MMI.Finite},
        AbstractVector{MMI.Continuous},
    },
    path="MLJFlux.LaplaceApproximation",
)
