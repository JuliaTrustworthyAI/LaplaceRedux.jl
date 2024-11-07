using Random: Random

"""
    toy_data_linear(N=100)

# Examples

```julia-repl
toy_data_linear()
```

"""
function toy_data_linear(N=100; seed=nothing)
    #set seed if available
    if seed !== nothing
        Random.seed!(seed)
    end
    # Number of points to generate.
    M = round(Int, N / 2)

    # Generate artificial data.
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i in 1:M])
    xt0s = Array([[x1s[i] - 5; x2s[i] - 5] for i in 1:M])

    # Store all the data for later.
    xs = [xt1s; xt0s]
    xs = map(x -> Float32.(x), xs)
    ts = [ones(M); zeros(M)]
    ts = map(x -> Float32.(x), ts)
    return xs, ts
end

"""
    toy_data_non_linear(N=100)

# Examples

```julia-repl
toy_data_non_linear()
```

"""
function toy_data_non_linear(N=100; seed=nothing)

    #set seed if available
    if seed !== nothing
        Random.seed!(seed)
    end

    # Number of points to generate.
    M = round(Int, N / 4)

    # Generate artificial data.
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i in 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i in 1:M]))

    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i in 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i in 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    xs = map(x -> Float32.(x), xs)
    ts = [ones(2 * M); zeros(2 * M)]
    ts = map(x -> Float32.(x), ts)
    return xs, ts
end

"""
    toy_data_multi(N=100)

# Examples

```julia-repl
toy_data_multi()
```

"""
function toy_data_multi(N=100; seed=nothing)

    #set seed if available
    if seed !== nothing
        Random.seed!(seed)
    end

    # Number of points to generate.
    M = round(Int, N / 4)

    # Generate artificial data.
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt1s = Array([[x1s[i] + 1; x2s[i] + 1] for i in 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt1s, Array([[x1s[i] - 7; x2s[i] - 7] for i in 1:M]))

    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt0s = Array([[x1s[i] + 1; x2s[i] - 7] for i in 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt0s, Array([[x1s[i] - 7; x2s[i] + 1] for i in 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    xs = map(x -> Float32.(x), xs)
    ts = [ones(M); ones(M) .* 2; ones(M) .* 3; ones(M) .* 4]
    ts = map(x -> Float32.(x), ts)
    return xs, ts
end

"""
    toy_data_regression(N=25, p=1; noise=0.3, fun::Function=f(x)=sin(2 * π * x))

A helper function to generate synthetic data for regression.
"""
function toy_data_regression(
    N=25,
    p=1;
    noise=0.3,
    fun::Function=f(x) = sin(x),
    xmax::AbstractFloat=8.0,
    center_origin=false,
    seed=nothing,
)
    if seed !== nothing
        Random.seed!(seed)
    end
    X = rand(N) * xmax
    X = center_origin ? X .- xmax / 2 : X
    X = Float32.(X)
    ε = randn(N) .* noise
    y = @.(fun(X)) + ε
    y = Float32.(y)
    return X, y
end
