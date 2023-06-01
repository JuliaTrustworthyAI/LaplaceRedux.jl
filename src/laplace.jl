using .Curvature
using Flux
using Flux.Optimise: Adam, update!
using Flux.Optimisers: destructure
using LinearAlgebra

mutable struct Laplace <: BaseLaplace
	model::Flux.Chain
	likelihood::Symbol
	subset_of_weights::Symbol
	subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}                   # indices of the subnetwork
	hessian_structure::Symbol
	curvature::Union{Curvature.CurvatureInterface,Nothing}
	σ::Real                                                                  # standard deviation in the Gaussian prior         
	μ₀::Real                                                                 # prior mean  
	μ::AbstractVector                                                        # posterior mean
	P₀::Union{AbstractMatrix,UniformScaling}                                 # prior precision (i.e. inverse covariance matrix)          
	H::Union{AbstractArray,Nothing}                                          # Hessian matrix 
	P::Union{AbstractArray,Nothing}                                          # posterior precision     
	Σ::Union{AbstractArray,Nothing}                                          # posterior covariance matrix
	n_params::Union{Int,Nothing}
	n_data::Union{Int,Nothing}
	n_out::Union{Int,Nothing}
	loss::Real
end

using Parameters

@with_kw struct LaplaceParams
	subset_of_weights::Symbol = :all
	subnetwork_indices::Union{Nothing,Vector{Vector{Int}}} = nothing
	hessian_structure::Symbol = :full
	backend::Symbol = :EmpiricalFisher
	σ::Real = 1.0
	μ₀::Real = 0.0
	λ::Real = 1.0                                                              # regularization parameter
	P₀::Union{Nothing,AbstractMatrix,UniformScaling} = nothing
	loss::Real = 0.0
end

"""
	Laplace(model::Any; loss_fun::Union{Symbol, Function}, kwargs...)    

Wrapper function to prepare Laplace approximation.
"""
function Laplace(model::Any; likelihood::Symbol, kwargs...)

	# Load hyperparameters:
	args = LaplaceParams(; kwargs...)

	# Assertions:
	@assert !(args.σ != 1.0 && likelihood != :regression) "Observation noise σ ≠ 1 only available for regression."
	@assert args.subset_of_weights ∈ [:all, :last_layer, :subnetwork] "`subset_of_weights` of weights should be one of the following: `[:all, :last_layer, :subnetwork]`"
	if (args.subset_of_weights == :subnetwork)
		@assert (args.subnetwork_indices !== nothing) "If `subset_of_weights` is `:subnetwork`, then `subnetwork_indices` should be a vector of vectors of integers."
		params = Flux.params(model)
		# Initialise a set of vectors 
		selected = Set{Vector{Int}}()
		for i in 1:length(args.subnetwork_indices)
			@assert !(args.subnetwork_indices[i] in selected) "Element $(i) in `subnetwork_indices` should be unique."
			@assert (args.subnetwork_indices[i][1] in 1:length(params)) "The first index of element $(i) in `subnetwork_indices` should be between 1 and $(length(params))."
			# Calculate numnber of dimensions of a parameter 
			n_dims = length(size(params[args.subnetwork_indices[i][1]]))
			@assert length(args.subnetwork_indices[i]) - 1 == n_dims "Element $(i) in `subnetwork_indices` should have $(n_dims) indices."
			for j in 2:length(args.subnetwork_indices[i])
				@assert (
					args.subnetwork_indices[i][j] in
					1:size(params[args.subnetwork_indices[i][1]], j-1)
				) "The index $(j) of element $(i) in `subnetwork_indices` should be between 1 and $(length(params[args.subnetwork_indices[i][1]][j-1]))."
			end
			push!(selected, args.subnetwork_indices[i])
		end
	end

	# Setup:
	P₀ = isnothing(args.P₀) ? UniformScaling(args.λ) : args.P₀
	nn = model
	n_out = outdim(nn)
	μ = reduce(vcat, [vec(θ) for θ in Flux.params(nn)])                       # μ contains the vertically concatenated parameters of the neural network

	# Instantiate LA:
	la = Laplace(
		model,
		likelihood,
		args.subset_of_weights,
		args.subnetwork_indices,
		args.hessian_structure,
		nothing,
		args.σ,
		args.μ₀,
		μ,
		P₀,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		n_out,
		args.loss,
	)

	params = get_params(la)

	# Instantiating curvature interface
	if la.subset_of_weights == :subnetwork
		la.curvature = getfield(Curvature, args.backend)(
			nn,
			likelihood,
			params,
			la.subset_of_weights,
			convert_subnetwork_indices(la.subnetwork_indices, params),
		)
	else
		la.curvature = getfield(Curvature, args.backend)(
			nn, likelihood, params, la.subset_of_weights, la.subnetwork_indices
		)
	end

	if la.subset_of_weights == :subnetwork
		la.n_params = length(la.subnetwork_indices)
	else
		la.n_params = length(reduce(vcat, [vec(θ) for θ in params]))                # number of params
	end
	la.μ = la.μ[(end - la.n_params + 1):end]                                    # adjust weight vector
	if typeof(la.P₀) <: UniformScaling
		la.P₀ = la.P₀(la.n_params)
	end

	# Sanity:
	if isa(la.P₀, AbstractMatrix)
		@assert all(size(la.P₀) .== la.n_params) "Dimensions of prior Hessian $(size(la.P₀)) do not align with number of parameters ($(la.n_params))"
	end

	return la
end

"""
	convert_subnetwork_indices(subnetwork_indices::AbstractArray)

Converts the subnetwork indices from the user given format [theta, row, column] to an Int i that corresponds to the index
of that weight in the flattened array of weights.
"""
function convert_subnetwork_indices(
	subnetwork_indices::Vector{Vector{Int}}, params::AbstractArray
)
	converted_indices = Vector{Int}()
	for i in subnetwork_indices
		Ca = reduce((acc, p) -> acc + length(p), params[1:(i[1] - 1)]; init=0)
		if length(i) == 2
			push!(converted_indices, Ca + i[2])
		elseif length(i) == 3
			push!(converted_indices, Ca + (i[2] - 1) * size(params[i[1]], 2) + i[3])
		end
	end
	return converted_indices
end

"""
	hessian_approximation(la::Laplace, d)

Computes the local Hessian approximation at a single data `d`.
"""
function hessian_approximation(la::Laplace, d)
	loss, H = getfield(Curvature, la.hessian_structure)(la.curvature, d)
	return loss, H
end

"""
	fit!(la::Laplace,data)

Fits the Laplace approximation for a data set.
The function returns the number of observations (n_data) that were used to update the Laplace object. 
It does not return the updated Laplace object itself because the function modifies the input Laplace object in place (as denoted by the use of '!' in the function's name).

# Examples

```julia-repl
using Flux, LaplaceRedux
x, y = LaplaceRedux.Data.toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn)
fit!(la, data)
```

"""
function fit!(la::Laplace, data; override::Bool=true)
	if override
		H = _init_H(la)
		loss = 0.0
		n_data = 0
	end

	# Training:
	for d in data
		loss_batch, H_batch = hessian_approximation(la, d)
		loss += loss_batch
		H += H_batch
		n_data += 1
	end

	# Store output:
	la.loss = loss                                                           # Loss
	la.H = H                                                                 # Hessian
	la.P = posterior_precision(la)                                           # posterior precision
	la.Σ = posterior_covariance(la)                                          # posterior covariance
	la.curvature.params = get_params(la)
	return la.n_data = n_data                                                # number of observations
end

"""
	glm_predictive_distribution(la::Laplace, X::AbstractArray)

Computes the linearized GLM predictive.
"""
function glm_predictive_distribution(la::Laplace, X::AbstractArray)
	𝐉, fμ = Curvature.jacobians(la.curvature, X)
	fvar = functional_variance(la, 𝐉)
	fvar = reshape(fvar, size(fμ)...)
	return fμ, fvar
end

"""
	functional_variance(la::Laplace,𝐉)

Compute the linearized GLM predictive variance as `𝐉ₙΣ𝐉ₙ'` where `𝐉=∇f(x;θ)|θ̂` is the Jacobian evaluated at the MAP estimate and `Σ = P⁻¹`.

"""
function functional_variance(la::Laplace, 𝐉)
	Σ = posterior_covariance(la)
	fvar = map(j -> (j' * Σ * j), eachcol(𝐉))
	return fvar
end

# Posterior predictions:
"""
	predict(la::Laplace, X::AbstractArray; link_approx=:probit)

Computes predictions from Bayesian neural network.
# Examples

```julia-repl
using Flux, LaplaceRedux
x, y = toy_data_linear()
data = zip(x,y)
nn = Chain(Dense(2,1))
la = Laplace(nn)
fit!(la, data)
predict(la, hcat(x...))
```

"""
function predict(la::Laplace, X::AbstractArray; link_approx=:probit)
	fμ, fvar = glm_predictive_distribution(la, X)

	# Regression:
	if la.likelihood == :regression
		return fμ, fvar
	end

	# Classification:
	if la.likelihood == :classification

		# Probit approximation
		if link_approx == :probit
			κ = 1 ./ sqrt.(1 .+ π / 8 .* fvar)
			z = κ .* fμ
		end

		if link_approx == :plugin
			z = fμ
		end

		# Sigmoid/Softmax
		if outdim(la) == 1
			p = Flux.sigmoid(z)
		else
			p = Flux.softmax(z; dims=1)
		end

		return p
	end
end

"""
	(la::Laplace)(X::AbstractArray; kwrgs...)

Calling a model with Laplace Approximation on an array of inputs is equivalent to explicitly calling the `predict` function.
"""
function (la::Laplace)(X::AbstractArray; kwrgs...)
	return predict(la, X; kwrgs...)
end

"""
	optimize_prior!(
		la::Laplace; 
		n_steps::Int=100, lr::Real=1e-1,
		λinit::Union{Nothing,Real}=nothing,
		σinit::Union{Nothing,Real}=nothing
	)
	
Optimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior!(
	la::Laplace;
	n_steps::Int=100,
	lr::Real=1e-1,
	λinit::Union{Nothing,Real}=nothing,
	σinit::Union{Nothing,Real}=nothing,
	verbose::Bool=false,
	tune_σ::Bool=la.likelihood == :regression,
)

	# Setup:
	logP₀ = isnothing(λinit) ? log.(unique(diag(la.P₀))) : log.([λinit])     # prior precision (scalar)
	logσ = isnothing(σinit) ? log.([la.σ]) : log.([σinit])                   # noise (scalar)
	opt = Adam(lr)                                                           # Adam is gradient descent (GD) optimization algorithm, using lr as learning rate
	show_every = round(n_steps / 10)
	i = 0
	if tune_σ
		@assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
		ps = Flux.params(logP₀, logσ)
	else
		if la.likelihood == :regression
			@warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
		end
		ps = Flux.params(logP₀)
	end
	loss(P₀, σ) = -log_marginal_likelihood(la; P₀=P₀[1], σ=σ[1])

	# Optimization:
	while i < n_steps
		gs = gradient(ps) do
			loss(exp.(logP₀), exp.(logσ))
		end
		update!(opt, ps, gs)                                                 # updates the values of the parameters using the calculated gradients and the Adam optimizer
		i += 1
		if verbose                                                           # if set to 'true', information about the optimization progress is printed out
			if i % show_every == 0
				@info "Iteration $(i): P₀=$(exp(logP₀[1])), σ=$(exp(logσ[1]))"
				@show loss(exp.(logP₀), exp.(logσ))
				println("Log likelihood: $(log_likelihood(la))")
				println("Log det ratio: $(log_det_ratio(la))")
				println("Scatter: $(_weight_penalty(la))")
			end
		end
	end

	# la.P = la.H + la.P₀
	# la.Σ = inv(la.P)

end
