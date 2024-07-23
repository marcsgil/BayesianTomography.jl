"""
    LinearInversion(povm, basis=gell_mann_matrices(size(first(povm), 1)))

Construct a linear inversion method for quantum state tomography.
"""
struct LinearInversion{T1,T2}
    C::Matrix{T1}
    pseudo_inv::Matrix{T1}
    basis::Array{T2,3}
    p_correction::Vector{T1}
    function LinearInversion(povm, basis=gell_mann_matrices(size(first(povm), 1), include_identity=false))
        C = stack(F -> real_orthogonal_projection(F, basis), povm, dims=1)
        pseudo_inv = C |> pinv
        T = eltype(pseudo_inv)
        p_correction = [real(tr(F)) / size(first(povm), 1) for F ∈ povm] |> vec
        new{T,complex(T)}(C, pseudo_inv, basis, p_correction)
    end
end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes, method::LinearInversion)
    d = Int(√(size(method.C, 2) + 1))
    xs = method.pseudo_inv * (vec(normalize(outcomes, 1)) - method.p_correction)
    project2density(linear_combination(xs, method.basis) + I / d), covariance(outcomes, method, xs)
end

function covariance(outcomes, method::LinearInversion, xs)
    N = sum(outcomes)
    C = method.C

    sum_residues = zero(eltype(C))

    @inbounds for i in axes(C, 1)
        temp = zero(sum_residues)

        temp += outcomes[i] / N - method.p_correction[i]
        @simd for j in axes(C, 2)
            temp -= C[i, j] * xs[j]
        end

        sum_residues += abs2(temp)
    end

    inv(C' * C) * sum_residues / (size(C, 1) - size(C, 2))
end