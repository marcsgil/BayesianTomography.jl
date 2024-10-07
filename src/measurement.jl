abstract type AbstractMeasurement end

struct Measurement{T1,T2} <: AbstractMeasurement
    measurement::T1
    traceless_part::Matrix{T2}
    trace_part::Vector{T2}
    dim::Int
end

function extract_trace(Π::AbstractMatrix, dim)
    real(tr(Π)) / dim
end

function extract_trace(ψ::AbstractVector, dim)
    sum(abs2, ψ) / dim
end



function set_decomposition!(traceless_part, trace_part, measurement, dim)
    for (Π, n, slice) ∈ zip(measurement, eachindex(trace_part), eachslice(traceless_part, dims=1))
        trace_part[n] = extract_trace(Π, dim)
        gell_mann_projection!(slice, Π)
    end
end

function set_decomposition!(traceless_part, trace_part, measurement, dim, tasks_per_thread)
    chunk_size = max(1, length(trace_part) ÷ (tasks_per_thread * nthreads()))
    chunk_traceless_part = partition(eachslice(traceless_part, dims=1), chunk_size)
    chunk_trace_part = partition(trace_part, chunk_size)
    chunk_measurements = partition(measurement, chunk_size)

    for iter ∈ zip(chunk_traceless_part, chunk_trace_part, chunk_measurements)
        fetch(@spawn set_decomposition!(iter..., dim))
    end
end

function get_decomposition(measurement)
    dim = size(first(measurement), 1)
    T = real(eltype(first(measurement)))
    traceless_part = Matrix{T}(undef, length(measurement), dim^2 - 1)
    trace_part = Vector{T}(undef, length(measurement))

    set_decomposition!(traceless_part, trace_part, measurement, dim)

    traceless_part, trace_part, dim
end

function Measurement(measurement)
    traceless_part, trace_part, dim = get_decomposition(measurement)
    Measurement{typeof(measurement),eltype(trace_part)}(measurement, traceless_part, trace_part, dim)
end

function get_probabilities!(dest, traceless_part, trace_part, θ)
    copy!(dest, trace_part)
    mul!(dest, traceless_part, θ, one(eltype(dest)), one(eltype(dest)))

    axpy!
end

function get_probabilities!(dest, measurement, θ)
    get_probabilities!(dest, measurement.traceless_part, measurement.trace_part, θ)
end

function get_probabilities(measurement, θ)
    dest = similar(measurement.trace_part)
    get_probabilities!(dest, measurement, θ)
    dest
end

function fisher!(F, T, probabilities)
    @tullio F[i, j] = T[k, i] * T[k, j] / probabilities[k]
end

function fisher!(F, probabilities, measurement::Measurement, θs)
    get_probabilities!(probabilities, measurement, θs)
    fisher!(F, measurement.traceless_part, probabilities)
    nothing
end

function fisher(measurement, θs)
    D = measurement.dim^2 - 1
    F = Matrix{eltype(θs)}(undef, D, D)
    probabilities = similar(measurement.trace_part)
    fisher!(F, probabilities, measurement, θs)
    F
end

"""
    kraus_transformation!(ρ, A)

Calculates A ρ A' and stores the result in ρ
"""
function kraus_transformation!(ρ::AbstractMatrix, A)
    rmul!(ρ, A')
    lmul!(A, ρ)
end

function kraus_transformation(ρ::AbstractMatrix, A)
    σ = copy(ρ)
    kraus_transformation!(σ, A)
    σ
end

function kraus_transformation(ψ::AbstractVector, A)
    A * ψ
end

function post_measurement_state!(ρ::AbstractMatrix, A)
    kraus_transformation!(ρ, A)
    ρ ./= tr(ρ)
    nothing
end

function post_measurement_state!(θ::AbstractVector, A)
    ρ = density_matrix_reconstruction(θ)
    post_measurement_state!(ρ, A)
    gell_mann_projection!(θ, ρ)
end

function post_measurement_state(state, A)
    result = copy(state)
    post_measurement_state!(result, A)
    result
end

struct ProportionalMeasurement{T1,T2} <: AbstractMeasurement
    effective_measurement::T1
    g::Matrix{T2}
    kraus_operator::UpperTriangular{T2,Matrix{T2}}
    inv_kraus_operator::UpperTriangular{T2,Matrix{T2}}
end

function get_g(measurement, ::Type{T}) where {T<:AbstractVector}
    sum(ψ -> ψ * ψ', measurement)
end

function get_g(measurement, ::Type{T}) where {T<:AbstractMatrix}
    sum(measurement)
end

function ProportionalMeasurement(measurement)
    g = get_g(measurement, typeof(first(measurement)))

    kraus_operator = cholesky(g).U
    inv_kraus_operator = inv(kraus_operator)

    effective_measurement = Measurement(kraus_transformation(Π, inv_kraus_operator') for Π ∈ measurement)

    T1 = typeof(effective_measurement)
    T2 = eltype(g)

    ProportionalMeasurement{T1,T2}(effective_measurement, g, kraus_operator, inv_kraus_operator)
end

function Base.getproperty(measurement::ProportionalMeasurement, symbol::Symbol)
    if symbol === :dim
        measurement.effective_measurement.dim
    elseif symbol === :traceless_part
        measurement.effective_measurement.traceless_part
    elseif symbol === :trace_part
        measurement.effective_measurement.trace_part
    elseif symbol === :measurement
        measurement.effective_measurement.measurement
    else
        getfield(measurement, symbol)
    end
end

function get_probabilities!(dest, measurement::ProportionalMeasurement, θ)
    η = post_measurement_state(θ, measurement.kraus_operator)
    get_probabilities!(dest, measurement.traceless_part, measurement.trace_part, η)
end

function fisher!(F, probabilities, measurement::ProportionalMeasurement, θs)
    σ = density_matrix_reconstruction(θs)
    A = measurement.kraus_operator
    kraus_transformation!(σ, A)
    N = real(tr(σ))

    D = measurement.dim^2 - 1
    J = Matrix{eltype(θs)}(undef, D, D)

    ωs = GellMannMatrices(measurement.dim, eltype(σ))
    Ωs = [kraus_transformation!(ω, A) for ω ∈ ωs]

    J = [real(tr(Ω * ω) / N - tr(σ * ω) * tr(Ω) / N^2) for ω ∈ ωs, Ω ∈ Ωs]

    get_probabilities!(probabilities, measurement, θs)
    fisher!(F, measurement.traceless_part, probabilities)

    F .= J' * F * J
    nothing
end

"""
    cond(povm::Union{AbstractArray{T},AbstractMatrix{T}}, p::Real=2) where {T<:AbstractMatrix}

Calculate the condition number of the linear transformation associated with the `povm`.
"""
function LinearAlgebra.cond(measurement, p::Real=2)
    cond(measurement.traceless_part, p)
end