"""
    LinearInversion(povm, basis=gell_mann_matrices(size(first(povm), 1)))

Tonstruct a linear inversion method for quantum state tomography.
"""
struct LinearInversion{T1,T2}
    T::Matrix{T1}
    pseudo_inv::Matrix{T1}
    basis::Array{T2,3}
    p_correction::Vector{T1}
end

function LinearInversion(povm, basis=gell_mann_matrices(size(first(povm), 1)))
    T = stack(Π -> real_orthogonal_projection(Π, (@view basis[:, :, begin+1:end])), povm, dims=1)
    pseudo_inv = T |> pinv
    p_correction = [real(tr(Π)) / size(first(povm), 1) for Π ∈ povm] |> vec
    LinearInversion(T, pseudo_inv, basis, p_correction)
end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes, method::LinearInversion{T1,T2}) where {T1,T2}
    d = size(method.basis, 1)
    θs = method.pseudo_inv * (vec(normalize(outcomes, 1)) - method.p_correction)
    linear_combination(vcat(convert(T1, 1 / √d), θs), method.basis), θs, covariance(outcomes, method, θs)
end

function covariance(outcomes, method::LinearInversion, θs)
    N = sum(outcomes)
    T = method.T

    sum_residues = zero(eltype(T))

    @inbounds for i in axes(T, 1)
        temp = zero(sum_residues)

        temp += outcomes[i] / N - method.p_correction[i]
        @simd for j in axes(T, 2)
            temp -= T[i, j] * θs[j]
        end

        sum_residues += abs2(temp)
    end

    inv(T' * T) * sum_residues / (size(T, 1) - size(T, 2))
end