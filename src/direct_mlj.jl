#module MLJLaplaceRedux
using Optimisers: Optimisers
using Flux
using Random
using Tables
using LinearAlgebra
using LaplaceRedux
using MLJBase
import MLJModelInterface as MMI
using Distributions: Normal

MLJBase.@mlj_model mutable struct LaplaceClassifier <: MLJBase.Probabilistic
    model::Union{Flux.Chain,Nothing} = nothing
    flux_loss = Flux.Losses.logitcrossentropy
    optimiser = Optimisers.Adam()
    epochs::Integer = 1000::(_ > 0)
    batch_size::Integer = 32::(_ > 0)
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    fit_prior_nsteps::Int = 100::(_ > 0)
    link_approx::Symbol = :probit::(_ in (:probit, :plugin))
end

MLJBase.@mlj_model mutable struct LaplaceRegressor <: MLJBase.Probabilistic
    model::Union{Flux.Chain,Nothing} = nothing
    flux_loss = Flux.Losses.mse
    optimiser = Optimisers.Adam()
    epochs::Integer = 1000::(_ > 0)
    batch_size::Integer = 32::(_ > 0)
    subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))
    subnetwork_indices = nothing
    hessian_structure::Union{HessianStructure,Symbol,String} =
        :full::(_ in (:full, :diagonal))
    backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))
    σ::Float64 = 1.0
    μ₀::Float64 = 0.0
    P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    fit_prior_nsteps::Int = 100::(_ > 0)
end

Laplace_Models = Union{LaplaceRegressor,LaplaceClassifier}

# for fit:
MMI.reformat(::Laplace_Models, X, y) = (MLJBase.matrix(X) |> permutedims, reshape(y, 1, :))
#for predict:
MMI.reformat(::Laplace_Models, X) = (MLJBase.matrix(X) |> permutedims,)

@doc """
    MMI.fit(m::Union{LaplaceRegressor,LaplaceClassifier}, verbosity, X, y)

Fit a Laplace model using the provided features and target values.

# Arguments
- `m::Laplace`: The Laplace (LaplaceRegressor or LaplaceClassifier) model to be fitted.
- `verbosity`: Verbosity level for logging.
- `X`: Input features, expected to be in a format compatible with MLJBase.matrix.
- `y`: Target values.

# Returns
- `fitresult`: a tuple (la,decode) cointaing  the fitted Laplace model and y[1],the first element of the categorical y vector.
- `cache`: a tuple containing a deepcopy of the model, the current state of the optimiser and the training loss history.
- `report`: A Namedtuple containing the loss history of the fitting process.
"""
function MMI.fit(m::Laplace_Models, verbosity, X, y)
    decode = y[1]

    if typeof(m) == LaplaceRegressor
        nothing
    else
        # Convert labels to integer format starting from 0 for one-hot encoding
        y_plain = MLJBase.int(y[1, :]) .- 1

        # One-hot encoding of labels
        unique_labels = unique(y_plain) # Ensure unique labels for one-hot encoding
        y = Flux.onehotbatch(y_plain, unique_labels) # One-hot encoding
    end

    # Make a copy of the model because Flux does not allow to mutate hyperparameters
    copied_model = deepcopy(m.model)

    data_loader = Flux.DataLoader((X, y); batchsize=m.batch_size)
    state_tree = Optimisers.setup(m.optimiser, copied_model)
    loss_history = []

    for epoch in 1:(m.epochs)
        loss_per_epoch = 0.0

        for (X_batch, y_batch) in data_loader
            # Forward pass: compute predictions
            y_pred = copied_model(X_batch)

            # Compute loss
            loss = m.flux_loss(y_pred, y_batch)

            # Compute gradients 
            grads, _ = gradient(copied_model, X_batch) do grad_model, X
                # Recompute predictions inside gradient context
                y_pred = grad_model(X)
                m.flux_loss(y_pred, y_batch)
            end

            # Update parameters using the optimizer and computed gradients
            state_tree, copied_model = Optimisers.update!(state_tree, copied_model, grads)

            # Accumulate the loss for this batch
            loss_per_epoch += sum(loss)  # Summing the batch loss
        end

        push!(loss_history, loss_per_epoch)

        # Print loss every 100 epochs if verbosity is 1 or more
        if verbosity >= 1 && epoch % 100 == 0
            println("Epoch $epoch: Loss: $loss_per_epoch ")
        end
    end

    la = LaplaceRedux.Laplace(
        copied_model;
        likelihood=:regression,
        subset_of_weights=m.subset_of_weights,
        subnetwork_indices=m.subnetwork_indices,
        hessian_structure=m.hessian_structure,
        backend=m.backend,
        σ=m.σ,
        μ₀=m.μ₀,
        P₀=m.P₀,
    )

    if typeof(m) == LaplaceClassifier
        la.likelihood = :classification
    end

    # fit the Laplace model:
    LaplaceRedux.fit!(la, data_loader)
    optimize_prior!(la; verbose=false, n_steps=m.fit_prior_nsteps)

    fitresult = (la, decode)
    report = (loss_history=loss_history,)
    cache = (deepcopy(m), state_tree, loss_history)
    return fitresult, cache, report
end

@doc """
    MMI.update(m::Union{LaplaceRegressor,LaplaceClassifier}, verbosity, X, y)

Update the Laplace model using the provided new data points.

# Arguments
- `m`: The Laplace (LaplaceRegressor or LaplaceClassifier) model to be fitted.
- `verbosity`: Verbosity level for logging.
- `X`: New input features, expected to be in a format compatible with MLJBase.matrix.
- `y`: New target values.

# Returns
- `fitresult`: a tuple (la,decode) cointaing  the updated fitted Laplace model and y[1],the first element of the categorical y vector.
- `cache`: a tuple containing a deepcopy of the model, the updated current state of the optimiser and training loss history.
- `report`: A Namedtuple containing the complete loss history of the fitting process.
"""
function MMI.update(m::Laplace_Models, verbosity, old_fitresult, old_cache, X, y)
    if typeof(m) == LaplaceRegressor
        nothing
    else
        # Convert labels to integer format starting from 0 for one-hot encoding
        y_plain = MLJBase.int(y[1, :]) .- 1

        # One-hot encoding of labels
        unique_labels = unique(y_plain) # Ensure unique labels for one-hot encoding
        y = Flux.onehotbatch(y_plain, unique_labels) # One-hot encoding
    end

    data_loader = Flux.DataLoader((X, y); batchsize=m.batch_size)
    old_model = old_cache[1]
    old_state_tree = old_cache[2]
    old_loss_history = old_cache[3]
    old_la = old_fitresult[1]

    epochs = m.epochs

    if MMI.is_same_except(m, old_model, :epochs)
        if epochs > old_model.epochs
            for epoch in (old_model.epochs + 1):(epochs)
                loss_per_epoch = 0.0

                for (X_batch, y_batch) in data_loader
                    # Forward pass: compute predictions
                    y_pred = old_la.model(X_batch)

                    # Compute loss
                    loss = m.flux_loss(y_pred, y_batch)

                    # Compute gradients 
                    grads, _ = gradient(old_la.model, X_batch) do grad_model, X
                        # Recompute predictions inside gradient context
                        y_pred = grad_model(X)
                        m.flux_loss(y_pred, y_batch)
                    end

                    # Update parameters using the optimizer and computed gradients
                    old_state_tree, old_la.model = Optimisers.update!(
                        old_state_tree, old_la.model, grads
                    )

                    # Accumulate the loss for this batch
                    loss_per_epoch += sum(loss)  # Summing the batch loss
                end

                push!(old_loss_history, loss_per_epoch)

                # Print loss every 100 epochs if verbosity is 1 or more
                if verbosity >= 1 && epoch % 100 == 0
                    println("Epoch $epoch: Loss: $loss_per_epoch ")
                end
            end

            la = LaplaceRedux.Laplace(
                old_la.model;
                likelihood=:regression,
                subset_of_weights=m.subset_of_weights,
                subnetwork_indices=m.subnetwork_indices,
                hessian_structure=m.hessian_structure,
                backend=m.backend,
                σ=m.σ,
                μ₀=m.μ₀,
                P₀=m.P₀,
            )
            if typeof(m) == LaplaceClassifier
                la.likelihood = :classification
            end

            # fit the Laplace model:
            LaplaceRedux.fit!(la, data_loader)
            optimize_prior!(la; verbose=false, n_steps=m.fit_prior_nsteps)

            fitresult = (la, y[1])
            report = (loss_history=old_loss_history,)
            cache = (deepcopy(m), old_state_tree, old_loss_history)

        else
            nothing
        end
    end

    if MMI.is_same_except(
        m,
        old_model,
        :fit_prior_nsteps,
        :subset_of_weights,
        :subnetwork_indices,
        :hessian_structure,
        :backend,
        :σ,
        :μ₀,
        :P₀,
    )
        println(" updating only the laplace optimization part")

        la = LaplaceRedux.Laplace(
            old_la.model;
            likelihood=:regression,
            subset_of_weights=m.subset_of_weights,
            subnetwork_indices=m.subnetwork_indices,
            hessian_structure=m.hessian_structure,
            backend=m.backend,
            σ=m.σ,
            μ₀=m.μ₀,
            P₀=m.P₀,
        )
        if typeof(m) == LaplaceClassifier
            la.likelihood = :classification
        end

        # fit the Laplace model:
        LaplaceRedux.fit!(la, data_loader)
        optimize_prior!(la; verbose=false, n_steps=m.fit_prior_nsteps)

        fitresult = la
        report = (loss_history=old_loss_history,)
        cache = (deepcopy(m), old_state_tree, old_loss_history)
    end

    return fitresult, cache, report
end

@doc """
    function MMI.is_same_except(m1::Laplace_Models, m2::Laplace_Models, exceptions::Symbol...) 

If both `m1` and `m2` are of `MLJType`, return `true` if the
following conditions all hold, and `false` otherwise:

- `typeof(m1) === typeof(m2)`

- `propertynames(m1) === propertynames(m2)`

- with the exception of properties listed as `exceptions` or bound to
  an `AbstractRNG`, each pair of corresponding property values is
  either "equal" or both undefined. (If a property appears as a
  `propertyname` but not a `fieldname`, it is deemed as always defined.)

The meaining of "equal" depends on the type of the property value:

- values that are themselves of `MLJType` are "equal" if they are
  equal in the sense of `is_same_except` with no exceptions.

- values that are not of `MLJType` are "equal" if they are `==`.

In the special case of a "deep" property, "equal" has a different
meaning; see [`deep_properties`](@ref)) for details.

If `m1` or `m2` are not `MLJType` objects, then return `==(m1, m2)`.

"""
function MMI.is_same_except(m1::Laplace_Models, m2::Laplace_Models, exceptions::Symbol...)
    typeof(m1) === typeof(m2) || return false
    names = propertynames(m1)
    propertynames(m2) === names || return false

    for name in names
        if !(name in exceptions)
            if !_isdefined(m1, name)
                !_isdefined(m2, name) || return false
            elseif _isdefined(m2, name)
                if name in deep_properties(LaplaceRegressor)
                    _equal_to_depth_one(getproperty(m1, name), getproperty(m2, name)) ||
                        return false
                else
                    (
                        is_same_except(getproperty(m1, name), getproperty(m2, name)) ||
                        getproperty(m1, name) isa AbstractRNG ||
                        getproperty(m2, name) isa AbstractRNG ||
                        (
                            getproperty(m1, name) isa Flux.Chain &&
                            getproperty(m2, name) isa Flux.Chain &&
                            _equal_flux_chain(getproperty(m1, name), getproperty(m2, name))
                        )
                    ) || return false
                end
            else
                return false
            end
        end
    end
    return true
end

# Define helper functions used in is_same_except
function _isdefined(obj, name)
    return hasproperty(obj, name)
end

function deep_properties(::Type)
    return Set{Symbol}()
end

function _equal_to_depth_one(a, b)
    return a == b
end

function _equal_flux_chain(chain1::Flux.Chain, chain2::Flux.Chain)
    if length(chain1.layers) != length(chain2.layers)
        return false
    end
    params1 = Flux.params(chain1)
    params2 = Flux.params(chain2)
    if length(params1) != length(params2)
        return false
    end
    for (p1, p2) in zip(params1, params2)
        if !isequal(p1, p2)
            return false
        end
    end
    for (layer1, layer2) in zip(chain1.layers, chain2.layers)
        if typeof(layer1) != typeof(layer2)
            return false
        end
    end
    return true
end

@doc """ 

 function  MMI.fitted_params(model::LaplaceRegressor, fitresult)
 
 
 This function extracts the fitted parameters from a `LaplaceRegressor` model.

 # Arguments
 - `model::LaplaceRegressor`: The Laplace regression model.
 - `fitresult`:  the Laplace approximation (`la`).

 # Returns
 A named tuple containing:
 - `μ`: The mean of the posterior distribution.
 - `H`: The Hessian of the posterior distribution.
 - `P`: The precision matrix of the posterior distribution.
 - `Σ`: The covariance matrix of the posterior distribution.
 - `n_data`: The number of data points.
 - `n_params`: The number of parameters.
 - `n_out`: The number of outputs.
 - `loss`: The loss value of the posterior distribution.

"""
function MMI.fitted_params(model::Laplace_Models, fitresult)
    la, decode = fitresult
    posterior = la.posterior
    return (
        μ=posterior.μ,
        H=posterior.H,
        P=posterior.P,
        Σ=posterior.Σ,
        n_data=posterior.n_data,
        n_params=posterior.n_params,
        n_out=posterior.n_out,
        loss=posterior.loss,
    )
end

@doc """
    MMI.training_losses(model::Union{LaplaceRegressor,LaplaceClassifier}, report)

Retrieve the training loss history from the given `report`.

# Arguments
- `model`: The model for which the training losses are being retrieved.
- `report`: An object containing the training report, which includes the loss history.

# Returns
- A collection representing the loss history from the training report.
"""
function MMI.training_losses(model::Laplace_Models, report)
    return report.loss_history
end

@doc """ 
function MMI.predict(m::LaplaceRegressor, fitresult, Xnew)

 Predicts the response for new data using a fitted Laplace  model.

 # Arguments
 - `m::LaplaceRegressor`: The Laplace  model.
 - `fitresult`: The result of the fitting procedure.
 - `Xnew`: The new data for which predictions are to be made.

 # Returns
    for LaplaceRegressor:
    - An array of Normal distributions, each centered around the predicted mean and variance for the corresponding input in `Xnew`.
    for LaplaceClassifier:
    - `MLJBase.UnivariateFinite`: The predicted class probabilities for the new data.
"""
function MMI.predict(m::Laplace_Models, fitresult, Xnew)
    la, decode = fitresult
    if typeof(m) == LaplaceRegressor
        yhat = LaplaceRedux.predict(la, Xnew; ret_distr=false)
        # Extract mean and variance matrices
        means, variances = yhat

        # Create Normal distributions from the means and variances
        return [Normal(μ, sqrt(σ)) for (μ, σ) in zip(means, variances)]

    else
        predictions =
            LaplaceRedux.predict(la, Xnew; link_approx=m.link_approx, ret_distr=false) |>
            permutedims

        return MLJBase.UnivariateFinite(MLJBase.classes(decode), predictions)
    end
end

MMI.metadata_pkg(
    LaplaceRegressor;
    name="LaplaceRedux",
    package_uuid="c52c1a26-f7c5-402b-80be-ba1e638ad478",
    package_url="https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl",
    is_pure_julia=true,
    is_wrapper=true,
    package_license="https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/blob/main/LICENSE",
)

MMI.metadata_pkg(
    LaplaceClassifier;
    name="LaplaceRedux",
    package_uuid="c52c1a26-f7c5-402b-80be-ba1e638ad478",
    package_url="https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl",
    is_pure_julia=true,
    is_wrapper=true,
    package_license="https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl/blob/main/LICENSE",
)

MLJBase.metadata_model(
    LaplaceClassifier;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite,MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Continuous), # table with mixed types
    },
    target_scitype=AbstractArray{<:MLJBase.Finite}, # ordered factor or multiclass
    supports_training_losses=true,
    load_path="LaplaceRedux.LaplaceClassifier",
)
# metadata for each model,
MLJBase.metadata_model(
    LaplaceRegressor;
    input_scitype=Union{
        AbstractMatrix{<:Union{MLJBase.Finite,MLJBase.Continuous}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Continuous), # table with mixed types
    },
    target_scitype=AbstractArray{MLJBase.Continuous},
    supports_training_losses=true,
    load_path="LaplaceRedux.LaplaceRegressor",
)

const DOC_LAPLACE_REDUX =
    "[Laplace Redux – Effortless Bayesian Deep Learning]" *
    "(https://proceedings.neurips.cc/paper/2021/hash/a3923dbe2f702eff254d67b48ae2f06e-Abstract.html), originally published in " *
    "Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., Hennig, P. (2021): \"Laplace Redux – Effortless Bayesian Deep Learning.\", NIPS'21: Proceedings of the 35th International Conference on Neural Information Processing Systems*, Article No. 1537, pp. 20089–20103"

"""
$(MMI.doc_header(LaplaceClassifier))

`LaplaceClassifier` implements the $DOC_LAPLACE_REDUX for classification models.

# Training data

In MLJ or MLJBase, given a dataset X,y and a Flux Chain adapt to the dataset, pass the chain to the model

laplace_model = LaplaceClassifier(model = Flux_Chain,kwargs...)

then bind an instance `laplace_model` to data with

    mach = machine(laplace_model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters (format: name-type-default value-restrictions)

- `model::Flux.Chain = nothing`:                                                               a Flux model provided by the user and compatible with the dataset.

- `flux_loss = Flux.Losses.logitcrossentropy` :                                                     a Flux loss function

- `optimiser = Adam()`                                                                              a Flux optimiser

- `epochs::Integer = 1000::(_ > 0)`:                                                                the number of training epochs.

- `batch_size::Integer = 32::(_ > 0)`:                                                              the batch size.

- `subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))`:                      the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.

- `subnetwork_indices = nothing`:                                                                   the indices of the subnetworks.

- `hessian_structure::Union{HessianStructure,Symbol,String} = :full::(_ in (:full, :diagonal))`:    the structure of the Hessian matrix, either `:full` or `:diagonal`.

- `backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))`:                                        the backend to use, either `:GGN` or `:EmpiricalFisher`.

- `σ::Float64 = 1.0`:                                                                               the standard deviation of the prior distribution.

- `μ₀::Float64 = 0.0`:                                                                              the mean of the prior distribution.

- `P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing`:                                     the covariance matrix of the prior distribution.

- `fit_prior_nsteps::Int = 100::(_ > 0) `:                                                          the number of steps used to fit the priors.

- `link_approx::Symbol = :probit::(_ in (:probit, :plugin))`:                                       the approximation to adopt to compute the probabilities.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.

- `training_losses(mach)`: return the loss history from report


# Fitted parameters

The fields of `fitted_params(mach)` are:

 - `μ`: The mean of the posterior distribution.

 - `H`: The Hessian of the posterior distribution.

 - `P`: The precision matrix of the posterior distribution.

 - `Σ`: The covariance matrix of the posterior distribution.

 - `n_data`: The number of data points.

 - `n_params`: The number of parameters.

 - `n_out`: The number of outputs.

 - `loss`: The loss value of the posterior distribution.



 # Report

The fields of `report(mach)` are:

- `loss_history`: an array containing the total loss per epoch.

# Accessor functions


# Examples

```
using MLJ
LaplaceClassifier = @load LaplaceClassifier pkg=LaplaceRedux

X, y = @load_iris

# Define the Flux Chain model
using Flux
model = Chain(
    Dense(4, 10, relu),
    Dense(10, 10, relu),
    Dense(10, 3)
)

#Define the LaplaceClassifier
model = LaplaceClassifier(model=model)

mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
training_losses(mach)      # loss history per epoch
pdf.(yhat, "virginica")    # probabilities for the "verginica" class
fitted_params(mach)        # NamedTuple with the fitted params of Laplace

```

See also [LaplaceRedux.jl](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl).

"""
LaplaceClassifier

"""
$(MMI.doc_header(LaplaceRegressor))

`LaplaceRegressor` implements the $DOC_LAPLACE_REDUX for regression models.

# Training data

In MLJ or MLJBase, given a dataset X,y and a Flux Chain adapt to the dataset, pass the chain to the model

laplace_model = LaplaceRegressor(model = Flux_Chain,kwargs...)

then bind an instance `laplace_model` to data with

    mach = machine(laplace_model, X, y)
where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyperparameters (format: name-type-default value-restrictions)

- `model::Flux.Chain = nothing`:                                                                    a Flux model provided by the user and compatible with the dataset.

- `flux_loss = Flux.Losses.logitcrossentropy` :                                                     a Flux loss function

- `optimiser = Adam()`                                                                              a Flux optimiser

- `epochs::Integer = 1000::(_ > 0)`:                                                                the number of training epochs.

- `batch_size::Integer = 32::(_ > 0)`:                                                              the batch size.

- `subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))`:                      the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.

- `subnetwork_indices = nothing`:                                                                   the indices of the subnetworks.

- `hessian_structure::Union{HessianStructure,Symbol,String} = :full::(_ in (:full, :diagonal))`:    the structure of the Hessian matrix, either `:full` or `:diagonal`.

- `backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))`:                                        the backend to use, either `:GGN` or `:EmpiricalFisher`.

- `σ::Float64 = 1.0`:                                                                               the standard deviation of the prior distribution.

- `μ₀::Float64 = 0.0`:                                                                              the mean of the prior distribution.

- `P₀::Union{AbstractMatrix,UniformScaling,Nothing} = nothing`:                                     the covariance matrix of the prior distribution.

- `fit_prior_nsteps::Int = 100::(_ > 0) `:                                                          the number of steps used to fit the priors.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.

- `training_losses(mach)`: return the loss history from report


# Fitted parameters

The fields of `fitted_params(mach)` are:

 - `μ`: The mean of the posterior distribution.

 - `H`: The Hessian of the posterior distribution.

 - `P`: The precision matrix of the posterior distribution.

 - `Σ`: The covariance matrix of the posterior distribution.

 - `n_data`: The number of data points.

 - `n_params`: The number of parameters.

 - `n_out`: The number of outputs.
 
 - `loss`: The loss value of the posterior distribution.


# Report

The fields of `report(mach)` are:

- `loss_history`: an array containing the total loss per epoch.




# Accessor functions



# Examples

```
using MLJ
using Flux
LaplaceRegressor = @load LaplaceRegressor pkg=LaplaceRedux
model = Chain(
    Dense(4, 10, relu),
    Dense(10, 10, relu),
    Dense(10, 1)
)
model = LaplaceRegressor(model=model)

X, y = make_regression(100, 4; noise=0.5, sparse=0.2, outliers=0.1)
mach = machine(model, X, y) |> fit!

Xnew, _ = make_regression(3, 4; rng=123)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
training_losses(mach)      # loss history per epoch
fitted_params(mach)        # NamedTuple with the fitted params of Laplace

```

See also [LaplaceRedux.jl](https://github.com/JuliaTrustworthyAI/LaplaceRedux.jl).

"""
LaplaceRegressor

#end # module
