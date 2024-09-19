struct StateTomographyProblem{T1<:AbstractFloat,T2<:Complex}
    measurement::Array{Matrix{T2}}
    dim::Int
    g::Matrix{T2}
    kraus_operator::UpperTriangular{T2,Matrix{T2}}
    inv_kraus_operator::UpperTriangular{T2,Matrix{T2}}
    effective_povm::Array{Matrix{T2}}
    traceless_part::Matrix{T1}
    trace_part::Vector{T1}
end

function StateTomographyProblem(measurement)
    dim = size(first(measurement), 1)

    g = sum(measurement)
    kraus_operator = cholesky(g).U
    inv_kraus_operator = inv(kraus_operator)

    effective_povm = deepcopy(measurement)
    for Π ∈ effective_povm
        kraus_transformation!(Π, inv_kraus_operator')
    end

    T = real(eltype(g))
    traceless_part = Matrix{T}(undef, length(measurement), dim^2 - 1)
    trace_part = Vector{T}(undef, length(measurement))

    for (n, Π) ∈ enumerate(effective_povm)
        trace_part[n] = real(tr(Π)) / dim
        gell_mann_projection!(view(traceless_part, n, :), Π)
    end

    StateTomographyProblem(measurement, dim, g, kraus_operator, inv_kraus_operator, effective_povm, traceless_part, trace_part)
end

"""
    kraus_transformation!(ρ, A)

Calculates A ρ A' and stores the result in ρ
"""
function kraus_transformation!(ρ, A)
    rmul!(ρ, A')
    lmul!(A, ρ)
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

function Base.show(io::IO, problem::StateTomographyProblem{T}) where {T}
    print(io, "$(problem.dim)-dimensional StateTomographyProblem{$T}")
end

function get_probabilities!(dest, traceless_part, trace_part, θ)
    copy!(dest, trace_part)
    mul!(dest, traceless_part, θ, one(eltype(dest)), one(eltype(dest)))
end

function get_probabilities!(dest, problem::StateTomographyProblem{T}, θ) where {T}
    η = post_measurement_state(θ, problem.kraus_operator)
    get_probabilities!(dest, problem.traceless_part, problem.trace_part, η)
end

function get_probabilities(problem::StateTomographyProblem{T}, θ) where {T}
    dest = Vector{T}(undef, size(problem.traceless_part, 1))
    get_probabilities!(dest, problem, θ)
    dest
end

function fisher!(F, T, probabilities)
    @tullio F[i, j] = T[k, i] * T[k, j] / probabilities[k]
end

function fisher!(F, probabilities, problem, θs)
    σ = density_matrix_reconstruction(θs)
    A = problem.kraus_operator
    kraus_transformation!(σ, A)
    N = real(tr(σ))

    D = problem.dim^2 - 1
    J = Matrix{eltype(θs)}(undef, D, D)

    ωs = GellMannMatrices(problem.dim, eltype(σ))
    Ωs = [kraus_transformation!(ω, A) for ω ∈ ωs]

    J = [real(tr(Ω * ω) / N - tr(σ * ω) * tr(Ω) / N^2) for ω ∈ ωs, Ω ∈ Ωs]

    get_probabilities!(probabilities, problem, θs)
    fisher!(F, problem.traceless_part, probabilities)

    F .= J' * F * J
    nothing
end

function fisher(problem, θs)
    D = problem.dim^2 - 1
    F = Matrix{eltype(θs)}(undef, D, D)
    probabilities = Vector{eltype(θs)}(undef, size(problem.traceless_part, 1))
    fisher!(F, probabilities, problem, θs)
    F
end

"""
    cond(povm::Union{AbstractArray{T},AbstractMatrix{T}}, p::Real=2) where {T<:AbstractMatrix}

Calculate the condition number of the linear transformation associated with the `povm`.
"""
function LinearAlgebra.cond(problem::StateTomographyProblem, p::Real=2)
    cond(problem.traceless_part, p)
end