function nth_off_diagonal(n)
    column = ceil(Int, (-1 + √(1 + 8n)) / 2)
    row = n - column * (column - 1) ÷ 2
    row, column + 1
end

struct GellMannMatrices{T<:Complex}
    dim::Int
end

function GellMannMatrices(dim, ::Type{T}=ComplexF32) where {T<:Complex}
    GellMannMatrices{T}(dim)
end

function Base.iterate(iter::GellMannMatrices{T}, state=1) where {T}
    dim = iter.dim
    state == dim^2 && return nothing

    result = zeros(T, (dim, dim))

    if state ≤ dim * (dim - 1) ÷ 2
        i, j = nth_off_diagonal(state)
        result[i, j] = 1 / √2
        result[j, i] = 1 / √2
    elseif state ≤ dim * (dim - 1)
        i, j = nth_off_diagonal(state - dim * (dim - 1) ÷ 2)
        result[i, j] = im / √2
        result[j, i] = -im / √2
    else
        j = state - dim * (dim - 1)
        factor = 1 / √(j * (j + 1))
        for k ∈ 1:j
            result[k, k] = factor
        end
        result[j+1, j+1] = -j * factor
    end

    result, state + 1
end

Base.IteratorSize(::GellMannMatrices) = Base.HasLength()
Base.IteratorEltype(::GellMannMatrices) = Base.HasEltype()
Base.eltype(::Type{GellMannMatrices{T}}) where {T} = Matrix{T}
Base.length(iter::GellMannMatrices) = iter.dim^2 - 1

function diag_retrival(n, θz)
    result = zero(eltype(θz))
    for j ∈ eachindex(θz)
        factor = if n ≤ j
            1
        elseif n == j + 1
            -j
        else
            0
        end

        result += θz[j] * factor / √(j^2 + j)
    end
    result
end

function gell_mann_projection!(θ::AbstractArray{T}, M) where {T}
    dim = size(M, 1)
    @assert length(θ) == dim^2 - 1

    sqrt2 = convert(T, sqrt(2))

    for n ∈ eachindex(θ)
        if n ≤ dim * (dim - 1) ÷ 2
            I = nth_off_diagonal(n)
            θ[n] = sqrt2 * real(M[I...])
        elseif n ≤ dim * (dim - 1)
            I = nth_off_diagonal(n - dim * (dim - 1) ÷ 2)
            θ[n] = sqrt2 * imag(M[I...])
        else
            j = n - dim * (dim - 1)
            θ[n] = (sum(j -> M[j, j], 1:j) - j * M[j+1, j+1]) / convert(T, √(j * (j + 1)))
        end
    end
end

function gell_mann_projection(M)
    dim = size(M, 1)
    θ = Vector{real(eltype(M))}(undef, dim^2 - 1)
    gell_mann_projection!(θ, M)
    θ
end

function gell_mann_reconstruction!(M, θ::AbstractVector{T}) where {T}
    dim = size(M, 1)
    @assert length(θ) == dim^2 - 1
    fill!(M, zero(T))

    inv_sqrt2 = convert(T, inv(sqrt(2)))

    n = 1
    # Off-diagonal elements (real part)
    for k ∈ 1:dim*(dim-1)÷2
        i, j = nth_off_diagonal(k)
        M[i, j] = θ[n] * inv_sqrt2
        M[j, i] = θ[n] * inv_sqrt2
        n += 1
    end

    # Off-diagonal elements (imaginary part)
    for k ∈ 1:dim*(dim-1)÷2
        i, j = nth_off_diagonal(k)
        M[i, j] += im * θ[n] * inv_sqrt2
        M[j, i] -= im * θ[n] * inv_sqrt2
        n += 1
    end

    θz = @view θ[dim*(dim-1)+1:end]
    # Diagonal elements
    for j in 1:dim
        M[j, j] = diag_retrival(j, θz)
    end
end

function gell_mann_reconstruction(θ)
    dims = Int(sqrt(length(θ) + 1))
    M = Matrix{complex(eltype(θ))}(undef, dims, dims)
    gell_mann_reconstruction!(M, θ)
    M
end

function density_matrix_reconstruction!(ρ, θ)
    gell_mann_reconstruction!(ρ, θ)
    dim = size(ρ, 1)
    factor = convert(eltype(ρ), 1 / dim)
    for i ∈ 1:dim
        ρ[i, i] += factor
    end
end

function density_matrix_reconstruction(θ)
    dims = Int(sqrt(length(θ) + 1))
    ρ = Matrix{complex(eltype(θ))}(undef, dims, dims)
    density_matrix_reconstruction!(ρ, θ)
    ρ
end