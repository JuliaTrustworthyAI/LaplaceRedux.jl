using Optimisers: Optimisers
using Flux
using Random
using Tables
using LinearAlgebra
using LaplaceRedux
using MLJBase: MLJBase
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
    observational_noise::Float64 = 1.0
    prior_mean::Float64 = 0.0
    prior_precision_matrix::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
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
    observational_noise::Float64 = 1.0
    prior_mean::Float64 = 0.0
    prior_precision_matrix::Union{AbstractMatrix,UniformScaling,Nothing} = nothing
    fit_prior_nsteps::Int = 100::(_ > 0)
end

LaplaceModels = Union{LaplaceRegressor,LaplaceClassifier}

# Aliases
const LM = LaplaceModels
function Base.getproperty(ce::LM, sym::Symbol)
    sym = sym === :σ ? :observational_noise : sym
    sym = sym === :μ₀ ? :prior_mean : sym
    sym = sym === :P₀ ? :prior_precision_matrix : sym
    return Base.getfield(ce, sym)
end
function Base.setproperty!(ce::LM, sym::Symbol, val)
    sym = sym === :σ ? :observational_noise : sym
    sym = sym === :μ₀ ? :prior_mean : sym
    sym = sym === :P₀ ? :prior_precision_matrix : sym
    return Base.setfield!(ce, sym, val)
end




# for fit:
function MMI.reformat(::LaplaceRegressor, X, y)
    return (MLJBase.matrix(X) |> permutedims, (reshape(y, 1, :), nothing))
end

function MMI.reformat(::LaplaceClassifier, X, y)
    X = MLJBase.matrix(X) |> permutedims
    y = MLJBase.categorical(y)
    labels = y.pool.levels
    y = Flux.onehotbatch(y, labels) # One-hot encoding

    return X, (y, labels)
end

MMI.reformat(::LaplaceModels, X) = (MLJBase.matrix(X) |> permutedims,)

MMI.selectrows(::LaplaceModels, I, Xmatrix, y) = (Xmatrix[:, I], (y[1][:, I], y[2]))
MMI.selectrows(::LaplaceModels, I, Xmatrix) = (Xmatrix[:, I],)



"""
    function dataset_shape(model::LaplaceRegression, X, y)

Compute the the number of features of the X input dataset and  the number of variables to predict from  the y  output dataset.

# Arguments
- `model::LaplaceModels`: The Laplace  model to fit.
- `X`: The input data for training.
- `y`: The target labels for training one-hot encoded.

# Returns
- (input size, output size)
"""
function dataset_shape(model::LaplaceModels, X, y)
    n_input = size(X, 1)
    dims = size(y)
    if length(dims) == 1
        n_output = 1
    else
        n_output = dims[1]
    end
    return (n_input, n_output)
end


"""
    default_build( seed::Int, shape)

Builds a default MLP Flux model compatible with the dimensions of the dataset, with reproducible initial weights.

# Arguments
- `seed::Int`: The seed for random number generation.
- `shape`: a tuple containing the dimensions of the input layer and the output layer.

# Returns
- The constructed Flux model, which consist in a simple MLP with 2 hidden layers with 20 neurons each and an input and output layers compatible with the dataset.
"""
function default_build(seed::Int, shape)
    Random.seed!(seed)
    (n_input, n_output) = shape
    
    chain = Chain(
        Dense(n_input, 20, relu),
        Dense(20, 20, relu),
        Dense(20, 20, relu),
        Dense(20, n_output)
    )
    
    return chain
end




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
function MMI.fit(m::LaplaceModels, verbosity, X, y)
    y, decode = y

    if (m.model === nothing)
        @warn "Warning: no Flux model has been provided in the model. LaplaceRedux will use a standard MLP with 2 hidden layers with 20 neurons each and input and output layers compatible with the dataset."
        shape = dataset_shape(m, X, y)

        m.model = default_build(11, shape)

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
        observational_noise=m.observational_noise,
        prior_mean=m.prior_mean,
        prior_precision_matrix=m.prior_precision_matrix,
    )

    if typeof(m) == LaplaceClassifier
        la.likelihood = :classification
    end

    # fit the Laplace model:
    LaplaceRedux.fit!(la, data_loader)
    optimize_prior!(la; verbosity= verbosity, n_steps=m.fit_prior_nsteps)

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
function MMI.update(m::LaplaceModels, verbosity, old_fitresult, old_cache, X, y)
    y_up, decode = y

    data_loader = Flux.DataLoader((X, y_up); batchsize=m.batch_size)
    old_model = old_cache[1]
    old_state_tree = old_cache[2]
    old_loss_history = old_cache[3]

    epochs = m.epochs

    if MMI.is_same_except(m, old_model, :epochs)
        old_la = old_fitresult[1]
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
                observational_noise=m.observational_noise,
                prior_mean=m.prior_mean,
                prior_precision_matrix=m.prior_precision_matrix,
            )
            if typeof(m) == LaplaceClassifier
                la.likelihood = :classification
            end

            # fit the Laplace model:
            LaplaceRedux.fit!(la, data_loader)
            optimize_prior!(la; verbosity = verbosity, n_steps=m.fit_prior_nsteps)

            fitresult = (la, decode)
            report = (loss_history=old_loss_history,)
            cache = (deepcopy(m), old_state_tree, old_loss_history)

        else
            println(
                "The number of epochs inserted is lower than the number of epochs already been trained. No update is necessary",
            )
            fitresult = (old_la, decode)
            report = (loss_history=old_loss_history,)
            cache = (deepcopy(m), old_state_tree, old_loss_history)
        end

    elseif MMI.is_same_except(
        m,
        old_model,
        :fit_prior_nsteps,
        :subset_of_weights,
        :subnetwork_indices,
        :hessian_structure,
        :backend,
        :observational_noise,
        :prior_mean,
        :prior_precision_matrix,
    )
        println(" updating only the laplace optimization part")
        old_la = old_fitresult[1]

        la = LaplaceRedux.Laplace(
            old_la.model;
            likelihood=:regression,
            subset_of_weights=m.subset_of_weights,
            subnetwork_indices=m.subnetwork_indices,
            hessian_structure=m.hessian_structure,
            backend=m.backend,
            observational_noise=m.observational_noise,
            prior_mean=m.prior_mean,
            prior_precision_matrix=m.prior_precision_matrix,
        )
        if typeof(m) == LaplaceClassifier
            la.likelihood = :classification
        end

        # fit the Laplace model:
        LaplaceRedux.fit!(la, data_loader)
        optimize_prior!(la; verbosity = verbosity, n_steps=m.fit_prior_nsteps)

        fitresult = (la, decode)
        report = (loss_history=old_loss_history,)
        cache = (deepcopy(m), old_state_tree, old_loss_history)

    end

    return fitresult, cache, report
end

@doc """
    function MMI.is_same_except(m1::LaplaceModels, m2::LaplaceModels, exceptions::Symbol...) 

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
meaning; see `MLJBase.deep_properties` for details.

If `m1` or `m2` are not `MLJType` objects, then return `==(m1, m2)`.

"""
function MMI.is_same_except(m1::LaplaceModels, m2::LaplaceModels, exceptions::Symbol...)
    typeof(m1) === typeof(m2) || return false
    names = propertynames(m1)
    propertynames(m2) === names || return false

    for name in names
        if !(name in exceptions)
            if !_isdefined(m1, name)
                !_isdefined(m2, name) || return false
            elseif _isdefined(m2, name)
                if name in MLJBase.deep_properties(LaplaceRegressor)
                    _equal_to_depth_one(getproperty(m1, name), getproperty(m2, name)) ||
                        return false
                else
                    (
                        MMI.is_same_except(getproperty(m1, name), getproperty(m2, name)) ||
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
function _isdefined(object, name)
    pnames = propertynames(object)
    fnames = fieldnames(typeof(object))
    name in pnames && !(name in fnames) && return true
    return isdefined(object, name)
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
 - `mean`: The mean of the posterior distribution.
 - `H`: The Hessian of the posterior distribution.
 - `P`: The precision matrix of the posterior distribution.
 - `cov_matrix`: The covariance matrix of the posterior distribution.
 - `n_data`: The number of data points.
 - `n_params`: The number of parameters.
 - `n_out`: The number of outputs.
 - `loss`: The loss value of the posterior distribution.

"""
function MMI.fitted_params(model::LaplaceModels, fitresult)
    la, decode = fitresult
    posterior = la.posterior
    return (
        mean=posterior.posterior_mean,
        H=posterior.H,
        P=posterior.P,
        cov_matrix=posterior.posterior_covariance_matrix,
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
function MMI.training_losses(model::LaplaceModels, report)
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
function MMI.predict(m::LaplaceModels, fitresult, Xnew)
    la, decode = fitresult
    if typeof(m) == LaplaceRegressor
        yhat = LaplaceRedux.predict(la, Xnew; ret_distr=false)
        # Extract mean and variance matrices
        means, variances = yhat

        # Create Normal distributions from the means and variances
        return vec([Normal(mean, sqrt(variance)) for (mean, variance) in zip(means, variances)])

    else
        predictions =
            LaplaceRedux.predict(la, Xnew; link_approx=m.link_approx, ret_distr=false) |>
            permutedims

        return MLJBase.UnivariateFinite(decode, predictions; pool=missing)
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
        AbstractMatrix{<:Union{MLJBase.Finite,MLJBase.Infinite}}, # matrix with mixed types
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
        AbstractMatrix{<:Union{MLJBase.Finite, MLJBase.Infinite}}, # matrix with mixed types
        MLJBase.Table(MLJBase.Finite, MLJBase.Infinite), # table with mixed types
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

In MLJ or MLJBase, given a dataset X,y and a `Flux_Chain` adapted to the dataset, pass the
chain to the model

```julia
laplace_model = LaplaceClassifier(model = Flux_Chain,kwargs...)
```

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

- `model::Union{Flux.Chain,Nothing} = nothing`:                                                     Either nothing or a Flux model provided by the user and compatible with the dataset. In the former case, LaplaceRedux will use a standard MLP with 2 hidden layers with 20 neurons each.

- `flux_loss = Flux.Losses.logitcrossentropy` :                                                     a Flux loss function

- `optimiser = Adam()`                                                                              a Flux optimiser

- `epochs::Integer = 1000::(_ > 0)`:                                                                the number of training epochs.

- `batch_size::Integer = 32::(_ > 0)`:                                                              the batch size.

- `subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))`:                      the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.

- `subnetwork_indices = nothing`:                                                                   the indices of the subnetworks.

- `hessian_structure::Union{HessianStructure,Symbol,String} = :full::(_ in (:full, :diagonal))`:    the structure of the Hessian matrix, either `:full` or `:diagonal`.

- `backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))`:                                        the backend to use, either `:GGN` or `:EmpiricalFisher`.

- `observational_noise (alias σ)::Float64 = 1.0`:                                                   the standard deviation of the prior distribution.

- `prior_mean (alias μ₀)::Float64 = 0.0`:                                                           the mean of the prior distribution.

- `prior_precision_matrix (alias P₀)::Union{AbstractMatrix,UniformScaling,Nothing} = nothing`:      the covariance matrix of the prior distribution.

- `fit_prior_nsteps::Int = 100::(_ > 0) `:                                                          the number of steps used to fit the priors.

- `link_approx::Symbol = :probit::(_ in (:probit, :plugin))`:                                       the approximation to adopt to compute the probabilities.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

 - `mean`: The mean of the posterior distribution.

 - `H`: The Hessian of the posterior distribution.

 - `P`: The precision matrix of the posterior distribution.

 - `cov_matrix`: The covariance matrix of the posterior distribution.

 - `n_data`: The number of data points.

 - `n_params`: The number of parameters.

 - `n_out`: The number of outputs.

 - `loss`: The loss value of the posterior distribution.



 # Report

The fields of `report(mach)` are:

- `loss_history`: an array containing the total loss per epoch.

# Accessor functions

- `training_losses(mach)`: return the loss history from report


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

In MLJ or MLJBase, given a dataset X,y and a `Flux_Chain` adapted to the dataset, pass the
chain to the model

```julia
laplace_model = LaplaceRegressor(model = Flux_Chain,kwargs...)
```

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

- `model::Union{Flux.Chain,Nothing} = nothing`:                                                     Either nothing or a Flux model provided by the user and compatible with the dataset. In the former case, LaplaceRedux will use a standard MLP with 2 hidden layers with 20 neurons each.
- `flux_loss = Flux.Losses.logitcrossentropy` :                                                     a Flux loss function

- `optimiser = Adam()`                                                                              a Flux optimiser

- `epochs::Integer = 1000::(_ > 0)`:                                                                the number of training epochs.

- `batch_size::Integer = 32::(_ > 0)`:                                                              the batch size.

- `subset_of_weights::Symbol = :all::(_ in (:all, :last_layer, :subnetwork))`:                      the subset of weights to use, either `:all`, `:last_layer`, or `:subnetwork`.

- `subnetwork_indices = nothing`:                                                                   the indices of the subnetworks.

- `hessian_structure::Union{HessianStructure,Symbol,String} = :full::(_ in (:full, :diagonal))`:    the structure of the Hessian matrix, either `:full` or `:diagonal`.

- `backend::Symbol = :GGN::(_ in (:GGN, :EmpiricalFisher))`:                                        the backend to use, either `:GGN` or `:EmpiricalFisher`.

- `observational_noise (alias σ)::Float64 = 1.0`:                                                   the standard deviation of the prior distribution.

- `prior_mean (alias μ₀)::Float64 = 0.0`:                                                           the mean of the prior distribution.

- `prior_precision_matrix (alias P₀)::Union{AbstractMatrix,UniformScaling,Nothing} = nothing`:      the covariance matrix of the prior distribution.

- `fit_prior_nsteps::Int = 100::(_ > 0) `:                                                          the number of steps used to fit the priors.


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

 - `mean`: The mean of the posterior distribution.

 - `H`: The Hessian of the posterior distribution.

 - `P`: The precision matrix of the posterior distribution.

 - `cov_matrix`: The covariance matrix of the posterior distribution.

 - `n_data`: The number of data points.

 - `n_params`: The number of parameters.

 - `n_out`: The number of outputs.
 
 - `loss`: The loss value of the posterior distribution.


# Report

The fields of `report(mach)` are:

- `loss_history`: an array containing the total loss per epoch.




# Accessor functions

- `training_losses(mach)`: return the loss history from report



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

