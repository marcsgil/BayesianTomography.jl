"""
    LinearInversion(problem::StateTomographyProblem)

Construct a linear inversion method for quantum state tomography.
"""
struct LinearInversion{T}
    problem::StateTomographyProblem{T}
    pseudo_inv::Matrix{T}
    θ_correction::Vector{T}
    function LinearInversion(problem::StateTomographyProblem{T}) where {T}
        pseudo_inv = pinv(problem.traceless_povm)
        θ_correction = pseudo_inv * problem.correction
        rmul!(θ_correction, -one(T))
        new{T}(problem, pseudo_inv, θ_correction)
    end
end

function prediction!(dest, probabilities::Array{Tp}, method::LinearInversion) where {Tp<:AbstractFloat}
    T = eltype(dest)
    copy!(dest, method.θ_correction)
    mul!(dest, method.pseudo_inv, probabilities, one(T), one(T))
end

function prediction!(dest, outcomes, method::LinearInversion)
    T = eltype(dest)
    N = convert(T, 1 / sum(outcomes))
    copy!(dest, method.θ_correction)
    mul!(dest, method.pseudo_inv, outcomes, N, one(T))
end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes, method::LinearInversion{T}) where {T}
    θs = Vector{T}(undef, size(method.problem.traceless_povm, 2))
    prediction!(θs, outcomes, method)
    density_matrix_reconstruction(θs), θs, covariance(outcomes, method, θs)
end

function covariance(outcomes, method::LinearInversion{T}, θs) where {T}
    """N = sum(outcomes)
    povm = method.povm

    sum_residues = zero(eltype(povm))

    @inbounds for i in axes(povm, 1)
        temp = zero(sum_residues)

        temp += outcomes[i] / N - method.correction[i]
        @simd for j in axes(povm, 2)
            temp -= povm[i, j] * θs[j]
        end

        sum_residues += abs2(temp)
    end

    inv(povm' * povm) * sum_residues / (size(povm, 1) - size(povm, 2))"""
    rand(T, method.problem.dim - 1, method.problem.dim - 1)
end