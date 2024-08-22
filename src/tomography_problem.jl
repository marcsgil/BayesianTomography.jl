struct StateTomographyProblem{T<:AbstractFloat}
    traceless_povm::Matrix{T}
    correction::Vector{T}
    dim::Int
    function StateTomographyProblem(traceless_povm::Matrix{T1}, correction::Vector{T2}) where {T1,T2}
        T = promote_type(T1, T2)
        dim = sqrt(size(traceless_povm, 2) + 1)
        @assert size(traceless_povm, 1) == length(correction)
        new{T}(traceless_povm, correction, Int(dim))
    end
end

function Base.show(io::IO, problem::StateTomographyProblem{T}) where {T}
    print(io, "$(problem.dim)-dimensional StateTomographyProblem{$T}")
end

function StateTomographyProblem(povm::Array{Matrix{T}}) where {T}
    rT = float(real(T))
    dim = size(first(povm), 1)
    M = length(povm)
    traceless_povm = Matrix{rT}(undef, M, dim^2 - 1)
    correction = Vector{rT}(undef, M)

    for (i, Π) in enumerate(povm)
        correction[i] = real(tr(Π)) / dim
        gell_mann_projection!(view(traceless_povm, i, :), Π)
    end

    StateTomographyProblem(traceless_povm, correction)
end

function get_probabilities!(dest, problem::StateTomographyProblem{T}, θ) where {T}
    copy!(dest, problem.correction)
    mul!(dest, problem.traceless_povm, θ, one(T), one(T))
end

function get_probabilities(problem::StateTomographyProblem{T}, θ) where {T}
    dest = Vector{T}(undef, size(problem.traceless_povm, 1))
    get_probabilities!(dest, problem, θ)
    dest
end

function fisher!(F, T, probabilities)
    #I = findall(x -> x > 0, vec(probs))
    #@tullio F[i, j] = T[I[k], i] * T[I[k], j] / probabilities[I[k]]
    @tullio F[i, j] := T[k, i] * T[k, j] / probabilities[k]
end

function fisher!(F, probabilities, problem, θs)
    get_probabilities!(probabilities, problem, θs)
    fisher!(F, T, probabilities)
end

function fisher(problem, θs)
    D = problem.dim^2 - 1
    F = Matrix{eltype(θs)}(undef, D, D)
    probabilities = Vector{eltype(θs)}(undef, size(problem.traceless_povm, 1))
    fisher!(F, probabilities, problem, θs)
    F
end

"""
    cond(povm::Union{AbstractArray{T},AbstractMatrix{T}}, p::Real=2) where {T<:AbstractMatrix}

Calculate the condition number of the linear transformation associated with the `povm`.
"""
function LinearAlgebra.cond(problem::StateTomographyProblem, p::Real=2)
    cond(problem.traceless_povm, p)
end