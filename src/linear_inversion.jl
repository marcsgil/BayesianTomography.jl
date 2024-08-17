"""
    LinearInversion(povm, basis=gell_mann_matrices(size(first(povm), 1)))

Construct a linear inversion method for quantum state tomography.
"""
struct LinearInversion{T1,T2}
    T::Matrix{T1}
    pseudo_inv::Matrix{T1}
    basis::Array{T2,3}
end

function LinearInversion(povm, basis=gell_mann_matrices(size(first(povm), 1)))
    T = stack(F -> real_orthogonal_projection(F, basis), povm, dims=1)
    pseudo_inv = pinv(T)
    T1 = eltype(T)
    T2 = complex(T1)
    LinearInversion{T1,T2}(T, pseudo_inv, basis)
end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes, method::LinearInversion)
    θs = method.pseudo_inv * vec(normalize(outcomes, 1))
    ρ = linear_combination(θs, method.basis)
    cov = covariance(outcomes, method, θs)
    ρ, θs, cov
end

function covariance(outcomes, method::LinearInversion, θs)
    N = sum(outcomes)
    _T = method.T
    T = @view _T[:, 2:end]
    p_correction = @view T[:, 1]


    sum_residues = zero(eltype(T))

    @inbounds for i in axes(T, 1)
        temp = zero(sum_residues)

        temp += outcomes[i] / N - p_correction[i]
        @simd for j in axes(T, 2)
            temp -= T[i, j] * θs[j]
        end

        sum_residues += abs2(temp)
    end

    inv(T' * T) * sum_residues / (size(T, 1) - size(T, 2))
end