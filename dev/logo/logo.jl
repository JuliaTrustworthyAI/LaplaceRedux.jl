using Pkg;
Pkg.activate("dev");

using Colors
using Flux
using LaplaceRedux
using LaplaceRedux.Data: toy_data_regression
using Luxor
using Statistics
using StatsBase: sample
using Random

julia_colors = Dict(
    :blue => Luxor.julia_blue,
    :red => Luxor.julia_red,
    :green => Luxor.julia_green,
    :purple => Luxor.julia_purple,
)
bg_color = RGBA(julia_colors[:blue]..., 0.25)

function get_data(N=500; seed=1234, σtrue=0.3)

    # Data:
    x, y = toy_data_regression(N; noise=σtrue, center_origin=true)
    xs = [[x] for x in x]
    X = permutedims(x)
    data = zip(xs, y)

    # Model:
    n_hidden = 50
    D = size(X, 1)
    nn = Chain(Dense(D, n_hidden, tanh), Dense(n_hidden, 1))
    loss(x, y) = Flux.Losses.mse(nn(x), y)

    # Training:
    opt = Adam(1e-3)
    epochs = 500
    avg_loss(data) = mean(map(d -> loss(d[1], d[2]), data))
    show_every = epochs / 10

    for epoch in 1:epochs
        for d in data
            gs = gradient(Flux.trainable(nn)) do
                l = loss(d...)
            end
            Flux.Optimise.update!(opt, Flux.trainable(nn), gs)
        end
        if epoch % show_every == 0
            println("Epoch " * string(epoch))
            @show avg_loss(data)
        end
    end

    # Laplace Approximation
    la = Laplace(nn; likelihood=:regression)
    fit!(la, data)
    optimize_prior!(la)

    return X, y, la
end

function logo_picture(;
    ndots=100,
    frame_size=500,
    ms=frame_size//100,
    mcolor=julia_colors[:red],
    interval_alpha=0.2,
    margin=-0.1,
    gt_color=julia_colors[:purple],
    gt_stroke_size=frame_size//50,
    m_alpha=0.5,
    seed=2022,
    σtrue=0.3,
    bg=true,
    bg_color="transparent",
    bg_border_color=julia_colors[:blue],
    n_steps=nothing,
)

    # Setup
    n_mcolor = length(mcolor)
    Random.seed!(seed)

    # Background 
    if bg
        circle(O, frame_size//2, :clip)
        setcolor(bg_color)
        box(Point(0, 0), frame_size, frame_size; action=:fill)
        setcolor(bg_border_color..., 1.0)
        circle(O, frame_size//2, :stroke)
    end

    # Data
    x, y, la = get_data(ndots; seed=seed, σtrue=σtrue)
    x_range = range(minimum(x) - 5; stop=maximum(x) + 5, length=50)
    fμ, fvar = la(permutedims(x_range))
    fμ = vec(fμ)
    fσ = vec(sqrt.(fvar))
    pred_std = sqrt.(fσ .^ 2 .+ la.prior.σ^2)
    y_lb = fμ .- 1.96 * pred_std
    y_ub = fμ .+ 1.96 * pred_std

    # Dots:
    idx = sample(1:length(x), ndots; replace=false)
    xplot, yplot = (x[idx], y[idx])
    _scale = (frame_size / (2 * maximum(abs.(collect(x_range))))) * (1 - margin)

    # Prediction interval:
    _order_lb = sortperm(collect(x_range))
    _order_ub = reverse(_order_lb)
    lb = [
        Point((_scale .* (x, y))...) for
        (x, y) in zip(collect(x_range)[_order_lb], y_lb[_order_lb])
    ]
    ub = [
        Point((_scale .* (x, y))...) for
        (x, y) in zip(collect(x_range)[_order_ub], y_ub[_order_ub])
    ]
    setcolor(sethue(gt_color)..., interval_alpha)
    poly(vcat(lb, ub); action=:fill)

    # Point predictions:
    setline(gt_stroke_size)
    setcolor(sethue(gt_color)..., 1.0)
    true_points = [Point((_scale .* (x, y))...) for (x, y) in zip(collect(x_range), fμ)]
    poly(true_points[1:(end - 1)]; action=:stroke)

    # Data
    data_plot = zip(xplot, yplot)
    for i in 1:length(data_plot)
        _x, _y = _scale .* collect(data_plot)[i]
        color_idx = i % n_mcolor == 0 ? n_mcolor : i % n_mcolor
        setcolor(mcolor..., m_alpha)
        circle(Point(_x, _y), ms; action=:fill)
    end
end

function draw_small_logo(
    filename="docs/src/assets/logo.svg", width=500; bg_color="transparent", kwrgs...
)
    frame_size = width
    Drawing(frame_size, frame_size, filename)
    if !isnothing(bg_color)
        background(bg_color)
    end
    origin()
    logo_picture(; kwrgs...)
    finish()
    return preview()
end

function animate_small_logo(
    filename="docs/src/assets/logo.gif", width=500; bg_color="transparent", kwrgs...
)
    frame_size = width
    anim = Movie(frame_size, frame_size, "logo", 1:10)
    function backdrop(scene, framenumber)
        return background(bg_color)
    end
    function frame(scene, framenumber)
        return logo_picture(;
            kwrgs...,
            m_alpha=1.0,
            switch_ce_color=false,
            bg_color=julia_colors[:blue],
            n_steps=framenumber,
        )
    end
    return animate(
        anim,
        [
            Scene(anim, backdrop, 1:10),
            Scene(anim, frame, 1:10; easingfunction=easeinoutcubic),
        ];
        creategif=true,
        pathname=filename,
    )
end

function draw_wide_logo(
    filename="docs/src/assets/wide_logo.png";
    _pkg_name="Laplace Redux",
    font_size=150,
    font_family="Tamil MN",
    font_fill=bg_color,
    font_color=Luxor.julia_blue,
    bg_color="transparent",
    picture_kwargs...,
)

    # Setup:
    height = Int(round(font_size * 2.4))
    fontsize(font_size)
    fontface(font_family)
    strs = split(_pkg_name)
    text_col_width = Int(round(maximum(map(str -> textextents(str)[3], strs)) * 1.05))
    width = Int(round(height + text_col_width))
    cw = [height, text_col_width]
    cells = Luxor.Table(height, cw)
    ms = Int(round(height / 10))
    gt_stroke_size = Int(round(height / 50))

    Drawing(width, height, filename)
    origin()
    background(bg_color)

    # Picture:
    @layer begin
        translate(cells[1])
        logo_picture(; frame_size=height, gt_stroke_size=gt_stroke_size, picture_kwargs...)
    end

    # Text:
    @layer begin
        translate(cells[2])
        fontsize(font_size)
        fontface(font_family)
        tiles = Tiler(cells.colwidths[2], height, length(strs), 1)
        for (pos, n) in tiles
            @layer begin
                translate(pos)
                setline(Int(round(gt_stroke_size / 5)))
                setcolor(font_fill)
                textoutlines(strs[n], O, :path; valign=:middle, halign=:center)
                fillpreserve()
                setcolor(font_color..., 1.0)
                strokepath()
            end
        end
    end

    finish()
    return preview()
end

draw_small_logo()
draw_wide_logo()
