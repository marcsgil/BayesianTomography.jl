"""
    LinearInversion(problem::StateTomographyProblem)

Construct a linear inversion method for quantum state tomography.
"""
struct LinearInversion end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes, measurement::Measurement, ::LinearInversion)
    θs = measurement.traceless_part \ (normalize(vec(outcomes), 1) .- measurement.trace_part)
    ρ = density_matrix_reconstruction(θs)
    project2density!(ρ)

    ρ, θs
end

function prediction(outcomes, measurement::ProportionalMeasurement, mthd; kwargs...)
    ρ, θs = prediction(outcomes, measurement.effective_measurement, mthd; kwargs...)

    post_measurement_state!(ρ, measurement.inv_kraus_operator)
    project2density!(ρ)
    gell_mann_projection!(θs, ρ)

    ρ, θs
end

struct PreAllocatedLinearInversion{T1<:AbstractMatrix,T2<:AbstractVector}
    pseudo_inv::T1
    θ_correction::T2
end

function PreAllocatedLinearInversion(measurement)
    pseudo_inv = pinv(measurement.traceless_part)
    θ_correction = pseudo_inv * measurement.trace_part
    PreAllocatedLinearInversion{typeof(pseudo_inv),typeof(θ_correction)}(pseudo_inv, θ_correction)
end

function prediction(outcomes, measurement::Measurement, method::PreAllocatedLinearInversion)
    θs = similar(method.θ_correction)
    T = eltype(θs)
    N = convert(T, 1 / sum(outcomes))
    copy!(θs, method.θ_correction)
    mul!(θs, method.pseudo_inv, vec(outcomes), N, -one(T))

    ρ = density_matrix_reconstruction(θs)
    project2density!(ρ)

    ρ, θs
end

struct NormalEquations{T1,T2}
    TdagT::T1
    Tdagq::T2
end

function NormalEquations(measurement)
    T = measurement.traceless_part
    TdagT = similar(T, size(T, 2), size(T, 2))
    Tdagq = similar(T, size(T, 2))
    NormalEquations{typeof(TdagT),typeof(Tdagq)}(TdagT, Tdagq)
end

function prediction(outcomes, measurement::Measurement, method::NormalEquations)
    mul!(method.TdagT, measurement.traceless_part', measurement.traceless_part)
    mul!(method.Tdagq, measurement.traceless_part', normalize(vec(outcomes), 1) .- measurement.trace_part)
    θs = method.TdagT \ method.Tdagq
    ρ = density_matrix_reconstruction(θs)
    project2density!(ρ)
    ρ, θs
end

function sum_residues(outcomes, method::LinearInversion, θs, N=one(T))
    probs = get_probabilities(method.problem, θs)
    mapreduce((x, y) -> (x * N - y)^2, +, outcomes, probs)
end

function covariance(outcomes, method::LinearInversion, θs, N=one(T))
    residues = sum_residues(outcomes, method, θs, N)
    trace_part = method.problem.trace_part
    inv(trace_part' * trace_part) * residues / (size(trace_part, 1) - size(trace_part, 2))
end