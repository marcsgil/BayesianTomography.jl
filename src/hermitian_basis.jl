function Z_matrix(j, d)
    @assert j ≤ d

    if j == 1
        return Matrix{ComplexF64}(I, d, d) / √d

    else
        diag = zeros(ComplexF64, d)
        for k ∈ 1:j-1
            diag[k] = 1
        end
        diag[j] = -j + 1
        result = diagm(diag)
        normalize!(result)
        result
    end
end

function W_matrix(j, d)
    result = zeros(ComplexF64, d, d)
    result[j, j] = 1
    result
end

function X_matrix(j, k, d)
    result = zeros(ComplexF64, d, d)
    result[j, k] = 1
    result[k, j] = 1
    normalize!(result)
    result
end

function Y_matrix(j, k, d)
    result = zeros(ComplexF64, d, d)
    result[j, k] = im
    result[k, j] = -im
    normalize!(result)
    result
end

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

function get_hermitian_basis(d; mode=:Z)
    @assert mode ∈ (:Z, :W)

    if mode == :Z
        diagonal = [Z_matrix(j, d) for j ∈ 1:d]
    elseif mode == :W
        diagonal = [W_matrix(j, d) for j ∈ 1:d]
    end

    Js = triangular_indices(d)
    Xs = [X_matrix(J..., d) for J ∈ Js]
    Ys = [Y_matrix(J..., d) for J ∈ Js]
    vcat(diagonal, Xs, Ys)
end

function real_representation(Ω, basis=get_hermitian_basis(size(Ω, 1)))
    [real(Ω ⋅ Ω′) for Ω′ ∈ basis]
end