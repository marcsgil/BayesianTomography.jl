function dict2array(dict::Dict{T1,T2}, size) where {T1<:Integer,T2<:Integer}
    result = Array{T2}(undef, size)
    for n ∈ eachindex(result)
        result[n] = get(dict, n, 0)
    end
    result
end

function array2dict(array::Array{T,N}) where {T,N}
    result = Dict{Int,T}()
    for (outcome, value) ∈ enumerate(array)
        if value != 0
            result[outcome] = value
        end
    end
    result
end

function history2array(history::AbstractArray{T,N}, size) where {T<:Integer,N}
    result = zeros(T, size)
    for outcome ∈ history
        result[outcome] += 1
    end
    result
end

function history2dict(history::AbstractArray{T,N}) where {T<:Integer,N}
    result = Dict{Int,T}()
    for outcome ∈ history
        result[outcome] = get(result, outcome, 0) + 1
    end
    result
end

function efective_povm(povm, observations)
    new_povm = Matrix{eltype(povm)}(undef, length(observations), size(povm, 2))
    new_obs = Vector{Float32}(undef, length(observations))

    for (n, pair) ∈ enumerate(observations)
        new_povm[n, :] = povm[pair.first, :]
        new_obs[n] = pair.second
    end

    new_povm, new_obs
end