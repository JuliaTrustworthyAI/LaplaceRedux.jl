# Stochastic Gradient Descent:
function sgd(X,y,∇,w_0,H_0,ρ_0=1.0,T=10000,ε=0.001)
    # Initialization:
    N = length(y)
    w_t = w_0 # initial parameters (prior mode)
    w_hat = 1/T * w_t # iterate averaging
    ρ = ρ_0 # initial step size
    t = 0 # iteration count
    while t<T
        n_t = rand(1:N) # sample minibatch (single sample)
        ρ = ρ * exp(-ε*t) # exponential decay 
        w_t = w_t - ρ .* ∇(w_t,w_0,X[n_t,:]',y[n_t],H_0) # update mode
        w_hat += 1/T * w_t # iterate averaging
        t += 1 # update count
    end
    return w_hat
end

# Newton's Method
function arminjo(𝓁, g_t, θ_t, d_t, args, ρ, c=1e-4)
    𝓁(θ_t .+ ρ .* d_t, args...) <= 𝓁(θ_t, args...) .+ c .* ρ .* d_t'g_t
end

function newton(𝓁, θ, ∇𝓁, ∇∇𝓁, args; max_iter=100, τ=1e-5)
    # Intialize:
    converged = false # termination state
    t = 1 # iteration count
    θ_t = θ # initial parameters
    # Descent:
    while !converged && t<max_iter 
        global g_t = ∇𝓁(θ_t, args...) # gradient
        global H_t = ∇∇𝓁(θ_t, args...) # hessian
        converged = all(abs.(g_t) .< τ) && isposdef(H_t) # check first-order condition
        # If not converged, descend:
        if !converged
            d_t = -inv(H_t)*g_t # descent direction
            # Line search:
            ρ_t = 1.0 # initialize at 1.0
            count = 1
            while !arminjo(𝓁, g_t, θ_t, d_t, args, ρ_t) 
                ρ_t /= 2
            end
            θ_t = θ_t .+ ρ_t .* d_t # update parameters
        end
        t += 1
    end
    # Output:
    return θ_t, H_t 
end