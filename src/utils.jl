using Random
function toy_data_linear(N=100)
    # Number of points to generate.
    M = round(Int, N / 2)
    Random.seed!(1234)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    xt0s = Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M])

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); zeros(M)];
    return xs, ts
end

using Random
function toy_data_non_linear(N=100)
    # Number of points to generate.
    M = round(Int, N / 4)
    Random.seed!(1234)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(2*M); zeros(2*M)];
    return xs, ts
end

# Plot data points:
using Plots
function plot_data!(plt,X,y)
    Plots.scatter!(plt, X[y.==1.0,1],X[y.==1.0,2], color=1, clim = (0,1), label="y=1")
    Plots.scatter!(plt, X[y.==0.0,1],X[y.==0.0,2], color=0, clim = (0,1), label="y=0")
end

# Plot contour of posterior predictive:
using Plots
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
        color=:plasma, legend=clegend, title=title, linewidth=0,
        xlim=xlim,
        ylim=ylim
    )
    plot_data!(plt,X,y)

end

# Helper function to predict from network trained for binary classification and producing logits as output:
predict(𝑴::Flux.Chain, X::AbstractArray) = Flux.σ.(𝑴(X))