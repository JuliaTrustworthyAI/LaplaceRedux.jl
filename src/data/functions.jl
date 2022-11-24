"""
    toy_data_linear(N=100)

# Examples

```julia-repl
toy_data_linear()
```

"""
function toy_data_linear(N=100)
    # Number of points to generate.
    M = round(Int, N / 2)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    xt0s = Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M])

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); zeros(M)];
    return xs, ts
end

"""
    toy_data_non_linear(N=100)

# Examples

```julia-repl
toy_data_non_linear()
```

"""
function toy_data_non_linear(N=100)
    # Number of points to generate.
    M = round(Int, N / 4)

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


"""
    toy_data_multi(N=100)

# Examples

```julia-repl
toy_data_multi()
```

"""
function toy_data_multi(N=100)
    # Number of points to generate.
    M = round(Int, N / 4)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 1; x2s[i] + 1] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt1s, Array([[x1s[i] - 7; x2s[i] - 7] for i = 1:M]))

    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt0s = Array([[x1s[i] + 1; x2s[i] - 7] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt0s, Array([[x1s[i] - 7; x2s[i] + 1] for i = 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); ones(M).*2; ones(M).*3; ones(M).*4];
    return xs, ts
end

"""
    toy_data_regression(N=25, p=1; noise=0.3, fun::Function=f(x)=sin(2 * π * x))

A helper function to generate synthetic data for regression.
"""
function toy_data_regression(N=25, p=1; noise=0.3, fun::Function=f(x)=sin(x), xmax::AbstractFloat=8.0)
    X = rand(N) * xmax
    ε = randn(N) .* noise
    y = @.(fun(X)) + ε
    return X, y
end