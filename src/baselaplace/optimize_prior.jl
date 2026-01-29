"""
    optimize_prior!(
        la::AbstractLaplace; 
        n_steps::Int=100, lr::Real=1e-1,
        λinit::Union{Nothing,Real}=nothing,
        σinit::Union{Nothing,Real}=nothing
    )
    
Optimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior!(
    la::AbstractLaplace;
    n_steps::Int=100,
    lr::Real=1e-1,
    λinit::Union{Nothing,Real}=nothing,
    σinit::Union{Nothing,Real}=nothing,
    verbosity::Int=0,
    tune_σ::Bool=la.likelihood == :regression,
)

    # Setup:
    logP₀ = isnothing(λinit) ? log.(unique(diag(la.prior.prior_precision_matrix))) : log.([λinit])      # prior precision (scalar)
    logσ = isnothing(σinit) ? log.([la.prior.observational_noise]) : log.([σinit])                      # noise (scalar)
    opt = Optimisers.Adam(lr)
    show_every = round(n_steps / 10)
    i = 0
    
    if tune_σ
        @assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
        params = (logP₀=logP₀, logσ=logσ)
    else
        if la.likelihood == :regression
            @warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
        end
        params = (logP₀=logP₀,)
    end
    
    # Initialize optimizer state
    opt_state = Optimisers.setup(opt, params)
    
    loss(P₀, σ) = -log_marginal_likelihood(la; P₀=P₀[1], σ=σ[1])

    # Optimization:
    while i < n_steps
        # Compute loss and gradients
        if tune_σ
            val, grads = Flux.withgradient(params) do p
                loss(exp.(p.logP₀), exp.(p.logσ))
            end
        else
            val, grads = Flux.withgradient(params) do p
                loss(exp.(p.logP₀), exp.(logσ))
            end
        end
        
        # Update parameters using new API
        opt_state, params = Optimisers.update(opt_state, params, grads[1])
        
        # Extract updated values
        logP₀ = params.logP₀
        if tune_σ
            logσ = params.logσ
        end
        
        i += 1
        if verbosity > 0
            if i % show_every == 0
                @info "Iteration $(i): P₀=$(exp(logP₀[1])), σ=$(exp(logσ[1]))"
                @show loss(exp.(logP₀), exp.(logσ))
                println("Log likelihood: $(log_likelihood(la))")
                println("Log det ratio: $(log_det_ratio(la))")
                println("Scatter: $(_weight_penalty(la))")
            end
        end
    end

    # Update the Laplace approximation object with optimized values
    la.prior.prior_precision_matrix = Diagonal(fill(exp(logP₀[1]), size(la.prior.prior_precision_matrix, 1)))
    if tune_σ
        la.prior.observational_noise = exp(logσ[1])
    end

    return la
end
