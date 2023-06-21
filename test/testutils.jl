using Flux, Plots, Random, Statistics, LaplaceRedux
using Flux.Optimise: update!, Adam
using CSV
using DataFrames
using JSON
using Serialization
using Tullio
using LinearAlgebra
using Zygote
using DelimitedFiles

Random.seed!(42)

function read_matrix_csv(filename::String)
    # Specify the file path relative to the current directory
    file_path = joinpath(@__DIR__, "datafiles", filename * ".csv")

    # Read the file using readdlm and create a Matrix{Float64}
    matrix_data = readdlm(file_path, ',', Float64)

    # Convert the matrix to a regular Julia matrix type
    matrix = Matrix{Float64}(matrix_data)

    return matrix
end

function rearrange_hessian(h::Matrix{Float64}, nn::Chain)
    to_row_order(h, nn) = h[gen_mapping_sq(Flux.params(nn))]

    return to_row_order(h, nn)
end

function rearrange_hessian_last_layer(h::Matrix{Float64}, nn::Chain)
    ps = [p for p in Flux.params(nn)]
    M = length(ps[end])
    N = length(ps[end - 1])
    return h[:, (end - M - N + 1):end][gen_mapping_sq(ps[(end - 1):end])]
end

function gen_mapping_sq(params)::Array{Tuple{Int64,Int64}}
    mapping_lin = gen_mapping(params)
    length_theta = sum(length, params)
    mapping_sq = Array{Tuple{Int64,Int64}}(undef, length_theta, length_theta)
    for (i, i_) in enumerate(mapping_lin)
        for (j, j_) in enumerate(mapping_lin)
            mapping_sq[i, j] = (i_, j_)
        end
    end
    return mapping_sq
end

import Base: getindex

function getindex(r::Matrix{Float64}, I::Matrix{Tuple{Int64,Int64}})
    l = Matrix{Float64}(undef, size(I))
    for (i, j) in Iterators.product(1:size(I, 1), 1:size(I, 2))
        # Unpack 2d index
        x, y = I[i, j]
        l[i, j] = r[x, y]
    end
    return l
end

function gen_mapping(params)
    theta_length = sum(length, params)
    offset = 0
    mapping = []
    for param in params
        indices = collect(1:length(param))
        indices_updated = vec(reshape(offset .+ indices, size(param))')
        append!(mapping, indices_updated)
        offset += length(param)
    end
    return mapping
end
