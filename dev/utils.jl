# Plot data points:
using Plots
"""
    plot_data!(plt,X,y)

# Examples

```julia-repl
using BayesLaplace, Plots
X, y = toy_data_linear(100)
plt = plot()
plot_data!(plt, hcat(X...)', y)
```

"""
function plot_data!(plt,X,y)
    Plots.scatter!(plt, X[y.==1.0,1],X[y.==1.0,2], color=1, clim = (0,1), label="y=1")
    Plots.scatter!(plt, X[y.==0.0,1],X[y.==0.0,2], color=0, clim = (0,1), label="y=0")
end

# Plot contour of posterior predictive:
using Plots
"""
    plot_contour(X,y,𝑴;clegend=true,title="",length_out=50,type=:laplace,zoom=0,xlim=nothing,ylim=nothing)

Generates a contour plot for the posterior predictive surface.  

# Examples

```julia-repl
using BayesLaplace, Plots
import BayesLaplace: predict
using NNlib: σ
X, y = toy_data_linear(100)
X = hcat(X...)'
β = [1,1]
𝑴 =(β=β,)
predict(𝑴, X) = σ.(𝑴.β' * X)
plot_contour(X, y, 𝑴)
```

"""
function plot_contour(X,y,𝑴;clegend=true,title="",length_out=50,type=:laplace,zoom=0,xlim=nothing,ylim=nothing)
    
    # Surface range:
    if isnothing(xlim)
        xlim = (minimum(X[:,1]),maximum(X[:,1])).+(zoom,-zoom)
    else
        xlim = xlim .+ (zoom,-zoom)
    end
    if isnothing(ylim)
        ylim = (minimum(X[:,2]),maximum(X[:,2])).+(zoom,-zoom)
    else
        ylim = ylim .+ (zoom,-zoom)
    end
    x_range = collect(range(xlim[1],stop=xlim[2],length=length_out))
    y_range = collect(range(ylim[1],stop=ylim[2],length=length_out))

    # Predictions:
    if type==:laplace
        Z = [predict(𝑴,[x, y])[1] for x=x_range, y=y_range]
    else
        Z = [plugin(𝑴,[x, y])[1] for x=x_range, y=y_range]
    end

    # Plot:
    plt = contourf(
        x_range, y_range, Z'; 
        legend=clegend, title=title, linewidth=0,
        xlim=xlim,
        ylim=ylim
    )
    plot_data!(plt,X,y)

end

# Helper function to predict from network trained for binary classification and producing logits as output:
import BayesLaplace: predict
predict(𝑴::Flux.Chain, X::AbstractArray) = Flux.σ.(𝑴(X))