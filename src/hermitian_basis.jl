"""
    triangular_indices(d)

Generate a vector of tuples representing the indices of the lower triangular part of a square matrix of dimension `d`.
"""
function triangular_indices(d)
    indices = Vector{Tuple{Int,Int}}(undef, d * (d - 1) ÷ 2)
    counter = 0
    for j ∈ 1:d
        for k ∈ 1:j-1
            counter += 1
            indices[counter] = (j, k)
        end
    end
    indices
end

"""
    X_matrix(j, k, d, ::Type{T}=ComplexF32) where {T<:Union{Real,Complex}}

Compute the real off diagonal matrix of the generalized Gell-Mann matrices in dimension `d`.

The type of the matrix elements is `T`, which defaults to `ComplexF32`.
The only non-zero elements are `X[j, k] = 1` and `X[k, j] = 1`.
The matrices are normalized to have unit Hilbert-Schmidt norm.

# Examples
```jldoctest
julia> X_matrix(1,2,2)
2×2 Matrix{ComplexF32}:
      0.0+0.0im  0.707107+0.0im
 0.707107+0.0im       0.0+0.0im
```
"""
function X_matrix(j, k, d, ::Type{T}=ComplexF32) where {T<:Union{Real,Complex}}
    result = zeros(T, d, d)
    result[j, k] = 1
    result[k, j] = 1
    normalize!(result)
    result
end

"""
    Y_matrix(j, k, d, ::Type{T}=ComplexF32) where {T<:Complex}

Compute the imaginary off diagonal matrix of the generalized Gell-Mann matrices in dimension `d`.

The type of the matrix elements is `T`, which defaults to `ComplexF32`.
The only non-zero elements are `Y[j, k] = im` and `Y[k, j] = -im`.
The matrices are normalized to have unit Hilbert-Schmidt norm.

# Examples
```jldoctest
julia> Y_matrix(1,2,2)
2×2 Matrix{ComplexF32}:
 0.0+0.0im       0.0+0.707107im
 0.0-0.707107im  0.0+0.0im
```
"""
function Y_matrix(j, k, d, ::Type{T}=ComplexF32) where {T<:Complex}
    result = zeros(T, d, d)
    result[j, k] = im
    result[k, j] = -im
    normalize!(result)
    result
end

"""
    Z_matrix(j, d, ::Type{T}=ComplexF32) where {T<:Union{Real,Complex}}

Compute the `j`'th diagonal matrix of the generalized Gell-Mann matrices in dimension `d`.

The type of the matrix elements is `T`, which defaults to `ComplexF32`.
The matrices are normalized to have unit Hilbert-Schmidt norm.
The identity matrix is returned when `j == 0`.

# Examples
```jldoctest
julia> Z_matrix(0, 3)
3×3 Matrix{ComplexF32}:
 0.57735+0.0im      0.0+0.0im      0.0+0.0im
     0.0+0.0im  0.57735+0.0im      0.0+0.0im
     0.0+0.0im      0.0+0.0im  0.57735+0.0im

julia> Z_matrix(1, 3)
3×3 Matrix{ComplexF32}:
 0.707107+0.0im        0.0+0.0im  0.0+0.0im
      0.0+0.0im  -0.707107+0.0im  0.0+0.0im
      0.0+0.0im        0.0+0.0im  0.0+0.0im

julia> Z_matrix(2, 3)
3×3 Matrix{ComplexF32}:
 0.408248+0.0im       0.0+0.0im        0.0+0.0im
      0.0+0.0im  0.408248+0.0im        0.0+0.0im
      0.0+0.0im       0.0+0.0im  -0.816497+0.0im
```
"""
function Z_matrix(j, d, ::Type{T}=ComplexF32) where {T<:Union{Real,Complex}}
    if j == 0
        result = Matrix{T}(I, d, d)
    else
        result = zeros(T, d, d)
        for k ∈ 1:j
            result[k, k] = 1
        end
        result[j+1, j+1] = -j
    end
    normalize!(result)
    result
end

"""
    gell_mann_matrices(d, ::Type{T}=ComplexF32; include_identity=true) where {T<:Complex}

Generate a set of Gell-Mann matrices of dimension `d`. 

The Gell-Mann matrices are a set of `d^2 - 1` linearly independent, traceless, 
Hermitian matrices that, when augmented with the identity,
form a basis for the space of `d × d` hermitian matrices.

The matrix order is real off-diagonal ([`X_matrix`](@ref)), 
imaginary off-diagonal ([`Y_matrix`](@ref)) and diagonal ([`Z_matrix`](@ref)).
The off-diagonal matrices follow the order given by [`triangular_indices`](@ref).

# Arguments
- `d`: The dimension of the Gell-Mann matrices.
- `include_identity`: A boolean flag indicating whether to include the identity matrix in the set. If this is `true`, the identity is the first element of the basis

# Returns
- A 3D array of Gell-Mann matrices. The last dimension is the index of the matrix in the basis.

# Examples
```jldoctest
julia> gell_mann_matrices(2,include_identity=false)
2×2×3 Array{ComplexF32, 3}:
[:, :, 1] =
      0.0+0.0im  0.707107+0.0im
 0.707107+0.0im       0.0+0.0im

[:, :, 2] =
 0.0+0.0im       0.0-0.707107im
 0.0+0.707107im  0.0+0.0im

[:, :, 3] =
 0.707107+0.0im        0.0+0.0im
      0.0+0.0im  -0.707107+0.0im
```
"""
function gell_mann_matrices(d, ::Type{T}=ComplexF32; include_identity=true) where {T<:Complex}
    f(x) = x + include_identity
    result = Array{T,3}(undef, d, d, d^2 - 1 + include_identity)

    if include_identity
        result[:, :, 1] .= Z_matrix(0, d, T)
    end

    for (n, J) ∈ enumerate(triangular_indices(d))
        result[:, :, f(n)] .= X_matrix(J..., d, T)
        result[:, :, f(n)+d*(d-1)÷2] .= Y_matrix(J..., d, T)
    end

    for j ∈ 1:d-1
        result[:, :, f(j)+d*(d-1)] .= Z_matrix(j, d, T)
    end

    result
end

"""
    basis_decomposition(Ω, basis=gell_mann_matrices(d))

Decompose the array `Ω` in the provided orthonormal basis.

If no basis is provided, the Gell-Mann matrices of appropriate dimension are used.

If `Ω` has dimension d, then `basis` should be an array with dimesnion `d+1` with the last 
dimension indexing the basis elements.
"""
function basis_decomposition(Ω, basis=gell_mann_matrices(d, eltype(Ω)))
    [real(Ω ⋅ Ω′) for Ω′ ∈ eachslice(basis, dims=ndims(basis))]
end