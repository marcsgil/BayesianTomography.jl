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

"""function prediction!(dest, probabilities::AbstractArray{Tp}, method::LinearInversion) where {Tp<:AbstractFloat}
    T = eltype(dest)
    copyto!(dest, method.θ_correction)
    mul!(dest, method.pseudo_inv, probabilities, one(T), one(T))
end"""

function prediction!(dest::AbstractArray{T}, outcomes, method::LinearInversion, N=one(T)) where {T}
    copy!(dest, method.θ_correction)
    mul!(dest, method.pseudo_inv, vec(outcomes), N, one(T))
end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes, method::LinearInversion{T}) where {T}
    θs = Vector{T}(undef, size(method.problem.traceless_povm, 2))
    N = convert(T, 1 / sum(outcomes))
    prediction!(θs, outcomes, method, N)

    ρ = density_matrix_reconstruction(θs)
    project2density!(ρ)
    gell_mann_projection!(θs, ρ)

    ρ, θs, covariance(outcomes, method, θs)
end

function sum_residues(outcomes, method::LinearInversion{T}, θs, N=one(T)) where {T}
    probs = get_probabilities(method.problem, θs)
    mapreduce((x, y) -> (x / N - y)^2, +, outcomes, probs)
end

function covariance(outcomes, method::LinearInversion{T}, θs, N=one(T)) where {T}
    residues = sum_residues(outcomes, method, θs, N)
    traceless_povm = method.problem.traceless_povm
    inv(traceless_povm' * traceless_povm) * residues / (size(traceless_povm, 1) - size(traceless_povm, 2))
end