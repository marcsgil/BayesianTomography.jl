function Z_matrix(j, d)
    @assert j ≤ d

    if j == 1
        return hermitianpart(Matrix{Float64}(I, d, d) / √d)

    else
        diag = zeros(d)
        for k ∈ 1:j-1
            diag[k] = 1
        end
        diag[j] = -j + 1
        result = diagm(diag)
        normalize!(result)
        return hermitianpart(result)
    end
end

function W_matrix(j, d)
    result = zeros(d, d)
    result[j, j] = 1
    hermitianpart(result)
end

function X_matrix(j, k, d)
    result = zeros(d, d)
    result[j, k] = 1
    result[k, j] = 1
    normalize!(result)
    hermitianpart(result)
end

function Y_matrix(j, k, d)
    result = zeros(ComplexF64, d, d)
    result[j, k] = im
    result[k, j] = -im
    normalize!(result)
    hermitianpart(result)
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
        diagonal = [W_matrix(j, d) for j ∈ 1:d-1]
    end

    Js = triangular_indices(d)
    Xs = [X_matrix(J..., d) for J ∈ Js]
    Ys = [Y_matrix(J..., d) for J ∈ Js]
    vcat(diagonal, Xs, Ys)
end

struct LinearInversion{T1,T2}
    pseudo_inv::Matrix{T1}
    basis::Vector{Matrix{T2}}
    function LinearInversion(povm)
        d = size(first(povm), 1)
        basis = get_hermitian_basis(d)
        A = [real(E ⋅ Ω) for E ∈ vec(povm), Ω ∈ vec(basis)]
        pseudo_inv = pinv(A)
        T = eltype(pseudo_inv)
        new{T,complex(T)}(pseudo_inv, basis)
    end
end

function prediction(outcomes, method::LinearInversion)
    vec_outcomes = vec(outcomes)
    xs = method.pseudo_inv * vec_outcomes
    ρ = sum(x * Ω for (x, Ω) ∈ zip(xs, method.basis)) |> project2density
end