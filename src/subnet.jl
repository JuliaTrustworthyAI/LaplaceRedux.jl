"""
validate_subnetwork_indices(
subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}, params
)

Determines whether subnetwork_indices is a valid input for specified parameters.
"""
function validate_subnetwork_indices(
    subnetwork_indices::Union{Nothing,Vector{Vector{Int}}}, params
)
    @assert (subnetwork_indices !== nothing) "If `subset_of_weights` is `:subnetwork`, then `subnetwork_indices` should be a vector of vectors of integers."
    # Check if subnetwork_indices is a vector containing an empty vector
    @assert !(subnetwork_indices == [[]]) "If `subset_of_weights` is `:subnetwork`, then `subnetwork_indices` should be a vector of vectors of integers."
    # Initialise a set of vectors
    selected = Set{Vector{Int}}()
    for (i, index) in enumerate(subnetwork_indices)
        @assert !(index in selected) "Element $(i) in `subnetwork_indices` should be unique."
        theta_index = index[1]
        @assert (theta_index in 1:length(params)) "The first index of element $(i) in `subnetwork_indices` should be between 1 and $(length(params))."
        # Calculate number of dimensions of a parameter
        theta_dims = size(params[theta_index])
        @assert length(index) - 1 == length(theta_dims) "Element $(i) in `subnetwork_indices` should have $(theta_dims) coordinates."
        for j in eachindex(index)[2:end]
            @assert (index[j] in 1:theta_dims[j - 1]) "The index $(j) of element $(i) in `subnetwork_indices` should be between 1 and $(theta_dims[j - 1])."
        end
        push!(selected, index)
    end
end

"""
convert_subnetwork_indices(subnetwork_indices::AbstractArray)

Converts the subnetwork indices from the user given format [theta, row, column] to an Int i that corresponds to the index
of that weight in the flattened array of weights.
"""
function convert_subnetwork_indices(
    subnetwork_indices::Vector{Vector{Int}}, params::AbstractArray
)
    converted_indices = Vector{Int}()
    for i in subnetwork_indices
        flat_theta_index = reduce((acc, p) -> acc + length(p), params[1:(i[1] - 1)]; init=0)
        if length(i) == 2
            push!(converted_indices, flat_theta_index + i[2])
        elseif length(i) == 3
            push!(
                converted_indices,
                flat_theta_index + (i[2] - 1) * size(params[i[1]], 2) + i[3],
            )
        end
    end
    return converted_indices
end
