"""
    reduced_representation(outcomes::Array{T,N}) where {T,N}

Converts a multi-dimensional array of outcomes into a 2D matrix in 
which the first row contains the indices of non-zero elements and the 
second row contains the corresponding non-zero values.

`outcomes` is a multi-dimensional array of outcomes where the `outcomes[n]` is 
the number of times the `n`-th outcome was observed.

The output is a matrix where the first row contains the indices of non-zero elements 
from the `outcomes` array and the second 
row contains the corresponding non-zero values.

This function has an inverse [`complete_representation`](@ref).

# Examples
```jldoctest
julia> outcomes = [0, 1, 0, 2, 0, 3]
6-element Vector{Int64}:
 0
 1
 0
 2
 0
 3

julia> reduced_representation(outcomes)
2×3 Matrix{Int64}:
 2  4  6
 1  2  3
```
"""
function reduced_representation(outcomes::Array{T,N}) where {T,N}
    n = count(x -> x != 0, outcomes)
    result = Matrix{T}(undef, 2, n)

    counter = 0
    for k ∈ eachindex(outcomes)
        if outcomes[k] != 0
            counter += 1
            result[1, counter] = k
            result[2, counter] = outcomes[k]
        end
    end

    result
end

"""
    complete_representation(outcomes::Matrix{T}, size) where {T}

Create a complete representation of the given outcomes.

`outcomes` is a matrix where the first row contains the indices of non-zero elements 
from of the complete representation and the second
row contains the corresponding non-zero values.

Returns a vector of size `size` where the i-th element is the value of the pair whose index is i in `outcomes`. 
If there is no such pair, the value is 0.

This function has an inverse [`reduced_representation`](@ref).

# Example
```jldoctest
julia> outcomes = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> complete_representation(outcomes, (2,2))
2×2 Matrix{Int64}:
 3  0
 4  0
```
"""
function complete_representation(outcomes::Matrix{T}, size) where {T}
    result = zeros(T, size)
    for pair ∈ eachslice(outcomes, dims=2)
        result[first(pair)] = last(pair)
    end
    result
end

"""
    History{T<:Integer}

A type that represents a history of outcomes.

# Fields
- `history::Vector{T}`: A vector of outcomes. `history[i]` is the outcome of the i-th measurement.
"""
struct History{T<:Integer}
    history::Vector{T}
end

"""
    reduced_representation(history::History)

Create a reduced representation of the given history.

# Arguments
- `history::History`: A History object which contains a history of events.

Return a matrix where each column is a pair (event, count). 
The event is the unique event from the history and count is the number of times the event has occurred.

# Example
```jldoctest
julia> h = History([1,1,1,2,1])
History{Int64}([1, 1, 1, 2, 1])

julia> reduced_representation(h)
2×2 Matrix{Int64}:
 2  1
 1  4
```
"""
function reduced_representation(history::History)
    cm = countmap(history.history)
    result = Matrix{eltype(history.history)}(undef, 2, length(cm))
    counter = 0
    for (key, value) ∈ cm
        counter += 1
        result[1, counter] = key
        result[2, counter] = value
    end
    result
end

"""
    complete_representation(history::History{T}, size) where {T}

Create a complete representation of the given history.

# Arguments
- `history::History`: A History object which contains a history of outcomes.
- `size`: The size of the resulting representation.

# Returns
- `result`: An array of size `size` where the i-th element is the number of times the i-th event occurred in the history.

# Example
```jldoctest
julia> h = History([1,1,1,2,1])
History{Int64}([1, 1, 1, 2, 1])

julia> complete_representation(h,(2,2))
2×2 Matrix{Int64}:
 4  0
 1  0
```
"""
function complete_representation(history::History{T}, size) where {T}
    result = zeros(T, size)
    for event ∈ history.history
        result[event] += one(T)
    end
    result
end