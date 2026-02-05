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
    tune_σ::Bool=(la.likelihood == :regression),
)

    # Setup:
    logP₀ = if isnothing(λinit)
        log.(unique(diag(la.prior.prior_precision_matrix)))   # prior precision (scalar)
    else
        log.([λinit])   # prior precision (scalar)
    end   # prior precision (scalar)
    logσ = isnothing(σinit) ? log.([la.prior.observational_noise]) : log.([σinit])                 # noise (scalar)
    opt_state_P = Optimisers.setup(Optimisers.Adam(lr), logP₀)
    opt_state_σ = Optimisers.setup(Optimisers.Adam(lr), logσ)
    show_every = round(n_steps / 10)
    i = 0
    if tune_σ
        @assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
    else
        if la.likelihood == :regression
            @warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
        end
    end
    loss(P₀, σ) = -log_marginal_likelihood(la; P₀=P₀[1], σ=σ[1])

    # Optimization:
    while i < n_steps
        if tune_σ
            gs_P, gs_σ = Flux.gradient((lp, ls) -> loss(exp.(lp), exp.(ls)), logP₀, logσ)
            opt_state_P, logP₀ = Optimisers.update!(opt_state_P, logP₀, gs_P)
            opt_state_σ, logσ = Optimisers.update!(opt_state_σ, logσ, gs_σ)
        else
            gs_P = Flux.gradient(lp -> loss(exp.(lp), exp.(logσ)), logP₀)[1]
            opt_state_P, logP₀ = Optimisers.update!(opt_state_P, logP₀, gs_P)
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
end
