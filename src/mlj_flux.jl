using Flux
using MLJFlux
import MLJModelInterface as MMI
using ProgressMeter
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
    P₀::Union{AbstractMatrix,UniformScaling}
    link_approx::Symbol
    fit_params::Dict{Symbol,Any}
    la::Union{Nothing,Laplace}
end

function LaplaceApproximation(
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
    P₀::Union{AbstractMatrix,UniformScaling}=UniformScaling(lambda),
    link_approx::Symbol=:probit,
    fit_params::Dict{Symbol,Any}=Dict{Symbol,Any}(:override => true),
) where {B,F,O,L}
    return LaplaceApproximation(
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
end

function MLJFlux.shape(model::LaplaceApproximation, X, y)
    X = X isa Matrix ? Tables.table(X) : X
    levels = MMI.classes(y[1])
    n_output = length(levels)
    n_input = length(Tables.schema(X).names)
    return (n_input, n_output)
end

function MLJFlux.build(model::LaplaceApproximation, rng, shape)
    chain = Flux.Chain(MLJFlux.build(model.builder, rng, shape...), model.finaliser)
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
    losses = (
        loss(chain(X[i]), y[i]) + penalty(parameters) / n_batches for i in 1:n_batches
    )
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

    if (model.batch_size > 1)
        fit!(la, DataLoader((X, y); batchsize=model.batch_size); model.fit_params...)
    else
        fit!(la, zip(X, y); model.fit_params...)
    end

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
    if model.batch_size < 0
        warning *= "Need `batch_size ≥ 0`. Resetting `batch_size = 1`. "
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
    X = MLJFlux.reformat(Xnew)
    println(X)
    probs = vcat([predict(la, MLJFlux.tomat(X[:, i]); link_approx=model.link_approx)' for i in 1:size(X, 2)]...)
    return MMI.UnivariateFinite(levels, probs)
end

MMI.metadata_model(
    LaplaceApproximation;
    input=Union{AbstractArray,MMI.Table(MMI.Continuous)},
    target=Union{AbstractArray{<:MMI.Finite},AbstractArray{<:MMI.Continuous}},
    path="MLJFlux.LaplaceApproximation",
)
