using Plots

function Plots.plot(
    la::Laplace,
    X::AbstractArray,
    y::AbstractArray;
    link_approx::Symbol=:probit,
    target::Union{Nothing,Real}=nothing,
    colorbar=true,
    title=nothing,
    length_out=50,
    zoom=-1,
    xlims=nothing,
    ylims=nothing,
    linewidth=0.1,
    lw=4,
    kwargs...,
)
    if la.likelihood == :regression
        @assert size(X, 1) == 1 "Cannot plot regression for multiple input variables."
    else
        @assert size(X, 1) == 2 "Cannot plot classification for more than two input variables."
    end

    if la.likelihood == :regression

        # REGRESSION

        # Surface range:
        if isnothing(xlims)
            xlims = (minimum(X), maximum(X)) .+ (zoom, -zoom)
        else
            xlims = xlims .+ (zoom, -zoom)
        end
        if isnothing(ylims)
            ylims = (minimum(y), maximum(y)) .+ (zoom, -zoom)
        else
            ylims = ylims .+ (zoom, -zoom)
        end
        x_range = range(xlims[1]; stop=xlims[2], length=length_out)
        y_range = range(ylims[1]; stop=ylims[2], length=length_out)

        title = isnothing(title) ? "" : title

        # Plot:
        scatter(
            vec(X),
            vec(y);
            label="ytrain",
            xlim=xlims,
            ylim=ylims,
            lw=lw,
            title=title,
            kwargs...,
        )
        _x = collect(x_range)[:, :]'
        fμ, fvar = la(_x)
        fμ = vec(fμ)
        fσ = vec(sqrt.(fvar))
        pred_std = sqrt.(fσ .^ 2 .+ la.σ^2)
        plot!(
            x_range,
            fμ;
            color=2,
            label="yhat",
            ribbon=(1.96 * pred_std, 1.96 * pred_std),
            lw=lw,
            kwargs...,
        )   # the specific values 1.96 are used here to create a 95% confidence interval
    else

        # CLASSIFICATION

        # Surface range:
        if isnothing(xlims)
            xlims = (minimum(X[1, :]), maximum(X[1, :])) .+ (zoom, -zoom)
        else
            xlims = xlims .+ (zoom, -zoom)
        end
        if isnothing(ylims)
            ylims = (minimum(X[2, :]), maximum(X[2, :])) .+ (zoom, -zoom)
        else
            ylims = ylims .+ (zoom, -zoom)
        end
        x_range = range(xlims[1]; stop=xlims[2], length=length_out)
        y_range = range(ylims[1]; stop=ylims[2], length=length_out)

        # Plot
        predict_ = function (X::AbstractVector)
            z = la(X; link_approx=link_approx)
            if outdim(la) == 1 # binary
                z = [1.0 - z[1], z[1]]
            end
            return z
        end
        Z = [predict_([x, y]) for x in x_range, y in y_range]
        Z = reduce(hcat, Z)
        if outdim(la) > 1
            if isnothing(target)
                @info "No target label supplied, using first."
            end
            target = isnothing(target) ? 1 : target
            title = isnothing(title) ? "p̂(y=$(target))" : title
        else
            target = isnothing(target) ? 2 : target
            title = isnothing(title) ? "p̂(y=$(target-1))" : title
        end

        # Contour:
        contourf(
            x_range,
            y_range,
            Z[Int(target), :];
            colorbar=colorbar,
            title=title,
            linewidth=linewidth,
            xlims=xlims,
            ylims=ylims,
            kwargs...,
        )
        # Samples:
        scatter!(X[1, :], X[2, :]; group=Int.(y), color=Int.(y), kwargs...)
    end
end
