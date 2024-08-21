using LinearAlgebra

function test_nth_off_diagonal()
    @assert nth_off_diagonal(1) == CartesianIndex(1, 2)
    @assert nth_off_diagonal(2) == CartesianIndex(1, 3)
    @assert nth_off_diagonal(3) == CartesianIndex(2, 3)
    @assert nth_off_diagonal(4) == CartesianIndex(1, 4)
    @assert nth_off_diagonal(5) == CartesianIndex(2, 4)
    @assert nth_off_diagonal(6) == CartesianIndex(3, 4)
    @assert nth_off_diagonal(7) == CartesianIndex(1, 5)
    @assert nth_off_diagonal(8) == CartesianIndex(2, 5)
    @assert nth_off_diagonal(9) == CartesianIndex(3, 5)
    @assert nth_off_diagonal(10) == CartesianIndex(4, 5)
end

function nth_off_diagonal(n)
    column = ceil(Int, (-1 + √(1 + 8n)) / 2)
    row = n - column * (column - 1) ÷ 2
    CartesianIndex(row, column + 1)
end

function gell_mann_projection!(θ::AbstractArray{T}, M) where {T}
    dim = size(M, 1)
    @assert length(θ) == dim^2 - 1

    inv_sqrt2 = convert(T, inv(sqrt(2)))

    for n ∈ eachindex(θ)
        if n ≤ dim * (dim - 1) ÷ 2
            I = nth_off_diagonal(n)
            θ[n] = inv_sqrt2 * real(M[I])
        elseif n ≤ dim * (dim - 1)
            I = nth_off_diagonal(n - dim * (dim - 1) ÷ 2)
            θ[n] = inv_sqrt2 * imag(M[I])
        else
            j = n - dim * (dim - 1)
            θ[n] = (sum(j -> M[j, j], 1:j) - j * M[j+1, j+1]) / convert(T, √(j * (j + 1)))
        end
    end
end

"""function vector_representation!(θ::AbstractArray{T}, M::Hermitian) where {T}
    dim = size(M, 1)
    @assert length(θ) == dim^2
    
    θ[1] = trace(M) / √dim
    gell_mann_projection!((@view θ[2:end]), M)
end"""

function gell_mann_reconstruction!(M, θ::AbstractVector{T}) where {T}
    dim = size(M, 1)
    @assert length(θ) == dim^2 - 1
    fill!(M, zero(T))

    sqrt2 = convert(T, sqrt(2))

    n = 1
    # Off-diagonal elements (real part)
    for j in 1:dim-1
        for i in j+1:dim
            M[i, j] = θ[n] * sqrt2
            M[j, i] = θ[n] * sqrt2
            n += 1
        end
    end

    # Off-diagonal elements (imaginary part)
    for j in 1:dim-1
        for i in j+1:dim
            M[i, j] -= im * θ[n] * sqrt2
            M[j, i] -= im * θ[n] * sqrt2
            n += 1
        end
    end

    # Diagonal elements
    for j in 1:dim-1
        factor = convert(T, sqrt(j * (j + 1)))
        diag_sum = sum(θ[k] * factor for k in dim*(dim-1)+1:dim*(dim-1)+j)
        for i in 1:j
            M[i, i] += diag_sum / factor
        end
        M[j+1, j+1] -= j * diag_sum / factor
    end
end

function density_operator_reconstruction!(ρ, θ)
    gell_mann_reconstruction!(ρ, θ)
    correction = 1 / size(ρ, 1)
    for n ∈ axes(ρ, 1)
        ρ[n, n] += correction
    end
end
##
ρ = hermitianpart(rand(ComplexF32, 2, 2))
ρ /= tr(ρ)
σ = Array{ComplexF32}(undef, size(ρ)...)
θ = Vector{Float32}(undef, length(ρ) - 1)

θ_representation!(θ, ρ)
density_operator_reconstruction!(σ, θ)

σ
ρ