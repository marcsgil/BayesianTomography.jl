"""
    LinearInversion(problem::StateTomographyProblem)

Construct a linear inversion method for quantum state tomography.
"""
struct LinearInversion{T}
    problem::StateTomographyProblem{T}
    pseudo_inv::Matrix{T}
    θ_correction::Vector{T}
    function LinearInversion(problem::StateTomographyProblem{T}) where {T}
        pseudo_inv = pinv(problem.traceless_part)
        θ_correction = pseudo_inv * problem.trace_part
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
    θs = Vector{T}(undef, size(method.problem.trace_part, 2))
    N = convert(T, 1 / sum(outcomes))
    prediction!(θs, outcomes, method, N)

    ρ = density_matrix_reconstruction(θs)
    project2density!(ρ)
    post_measurement_state!(ρ, method.problem.inv_kraus_operator)
    gell_mann_projection!(θs, ρ)

    ρ, θs, covariance(outcomes, method, θs, N)
end

function sum_residues(outcomes, method::LinearInversion{T}, θs, N=one(T)) where {T}
    probs = get_probabilities(method.problem, θs)
    mapreduce((x, y) -> (x * N - y)^2, +, outcomes, probs)
end

function covariance(outcomes, method::LinearInversion{T}, θs, N=one(T)) where {T}
    residues = sum_residues(outcomes, method, θs, N)
    trace_part = method.problem.trace_part
    inv(trace_part' * trace_part) * residues / (size(trace_part, 1) - size(trace_part, 2))
end