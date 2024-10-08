abstract type AbstractMeasurement{T<:AbstractFloat,TM<:AbstractMatrix{T}} end

struct Measurement{T<:AbstractFloat,TM<:AbstractMatrix{T}} <: AbstractMeasurement{T,TM}
    measurement_matrix::TM
    dim::Int
end

function Measurement(itr)
    dim = size(first(itr), 1)
    measurement_matrix = hcat((get_coefficients(Π) for Π ∈ itr)...)'
    Measurement(measurement_matrix, dim)
end

get_traceless_part(measurement_matrix::AbstractMatrix) = @view measurement_matrix[:, begin+1:end]
get_trace_part(measurement_matrix::AbstractMatrix) = @view measurement_matrix[:, begin]
get_traceless_part(μ) = get_traceless_part(μ.measurement_matrix)
get_trace_part(μ) = get_trace_part(μ.measurement_matrix)

function get_probabilities!(dest, measurement_matrix, θ)
    dim = Int(sqrt(size(measurement_matrix, 2)))
    trace_part = get_trace_part(measurement_matrix)
    traceless_part = get_traceless_part(measurement_matrix)

    copy!(dest, trace_part)
    mul!(dest, traceless_part, θ, one(eltype(dest)), convert(eltype(dest), 1 / √dim))
end

function get_probabilities!(dest, μ::Measurement, θ)
    get_probabilities!(dest, μ.measurement_matrix, θ)
end

function get_probabilities(μ, θ)
    dest = similar(μ.measurement_matrix, size(μ.measurement_matrix, 1))
    get_probabilities!(dest, μ, θ)
    dest
end

function fisher!(F, T, probabilities)
    @tullio F[i, j] = T[k, i] * T[k, j] / probabilities[k]
end

function fisher!(F, probabilities, μ::Measurement, θs)
    get_probabilities!(probabilities, μ, θs)
    fisher!(F, get_traceless_part(μ), probabilities)
    nothing
end

function fisher(measurement, θs)
    D = measurement.dim^2 - 1
    F = Matrix{eltype(θs)}(undef, D, D)
    probabilities = similar(μ.measurement_matrix, size(μ.measurement_matrix, 1))
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

struct ProportionalMeasurement{T<:AbstractFloat,TM<:AbstractMatrix{T}} <: AbstractMeasurement{T,TM}
    measurement_matrix::TM
    dim::Int
    g::Matrix{Complex{T}}
    kraus_operator::UpperTriangular{Complex{T},Matrix{Complex{T}}}
    inv_kraus_operator::UpperTriangular{Complex{T},Matrix{Complex{T}}}
end

function get_g(measurement, ::Type{T}) where {T<:AbstractVector}
    sum(ψ -> ψ * ψ', measurement)
end

function get_g(measurement, ::Type{T}) where {T<:AbstractMatrix}
    sum(measurement)
end

function ProportionalMeasurement(itr)
    g = get_g(itr, typeof(first(itr)))

    kraus_operator = cholesky(g).U
    inv_kraus_operator = inv(kraus_operator)

    dim = size(first(itr), 1)
    measurement_matrix = hcat((get_coefficients(kraus_transformation(Π, inv_kraus_operator')) for Π ∈ itr)...)'

    ProportionalMeasurement(measurement_matrix, dim, g, kraus_operator, inv_kraus_operator)
end

function get_probabilities!(dest, μ::ProportionalMeasurement, θ)
    η = post_measurement_state(θ, μ.kraus_operator)
    get_probabilities!(dest, μ.measurement_matrix, η)
end

function fisher!(F, probabilities, μ::ProportionalMeasurement, θs)
    σ = density_matrix_reconstruction(θs)
    A = μ.kraus_operator
    kraus_transformation!(σ, A)
    N = real(tr(σ))

    D = μ.dim^2 - 1
    J = Matrix{eltype(θs)}(undef, D, D)

    ωs = GellMannMatrices(μ.dim, eltype(σ))
    Ωs = [kraus_transformation!(ω, A) for ω ∈ ωs]

    J = [real(tr(Ω * ω) / N - tr(σ * ω) * tr(Ω) / N^2) for ω ∈ ωs, Ω ∈ Ωs]

    get_probabilities!(probabilities, μ, θs)
    fisher!(F, μ.traceless_part, probabilities)

    F .= J' * F * J
    nothing
end

"""
    cond(povm::Union{AbstractArray{T},AbstractMatrix{T}}, p::Real=2) where {T<:AbstractMatrix}

Calculate the condition number of the linear transformation associated with the `povm`.
"""
function LinearAlgebra.cond(μ, p::Real=2)
    cond(get_traceless_part(μ), p)
end