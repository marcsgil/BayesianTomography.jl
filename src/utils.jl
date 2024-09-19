"""
    simulate_outcomes(ψ::AbstractVector, measurement, N)
    simulate_outcomes(ρ::AbstractMatrix, measurement, N)
    simulate_outcomes(probs, N)

Simulate the `N` outcomes of a quantum `measurement` on a quantum state.

The state can be pure or mixed, and it is represented by a vector `ψ` or a density matrix `ρ`, respectively.
Alternativelly, one can directly provide the probabilities of the outcomes in the `probs` array.
"""
function simulate_outcomes(ψ::AbstractVector, measurement, N)
    probs = [real(dot(ψ, E, ψ)) for E in measurement]
    simulate_outcomes(probs, N)
end

function simulate_outcomes(ρ::AbstractMatrix, povm, N)
    probs = [real(ρ ⋅ E) for E in povm]
    simulate_outcomes(probs, N)
end

function simulate_outcomes(probs, N)
    outcomes = copy(probs)
    simulate_outcomes!(outcomes, N)
    outcomes
end


"""
    simulate_outcomes!(probs, N; atol=1e-3)

Simulate the `N` outcomes of a probability specified by the `probs` array.
The results are stored in the `probs` array.
"""
function simulate_outcomes!(probs, N)
    dist = Categorical(map(x -> x > 0 ? x : zero(x), normalize(vec(probs), 1)))
    samples = rand(dist, N)

    Threads.@threads for n in eachindex(probs)
        probs[n] = count(x -> x == n, samples)
    end
end

"""
    fidelity(ρ::AbstractMatrix, σ::AbstractMatrix)
    fidelity(ψ::AbstractVector, φ::AbstractVector)

Calculate the fidelity between two quantum states.

The states can be pure or mixed, and they are represented by vectors `ψ` and `φ` or density matrices `ρ` and `σ`, respectively.
"""
function fidelity(ρ::AbstractMatrix, σ::AbstractMatrix)
    abs2(tr(sqrt(ρ * σ)))
end

fidelity(ρ, σ) = fidelity(σ, ρ)

fidelity(ψ::AbstractVector, φ::AbstractVector) = abs2(ψ ⋅ φ) / sum(abs2, ψ) / sum(abs2, φ)

function get_w(λs, j)
    (sum(view(λs, 1:j)) - 1) / j
end

function project_onto_simplex!(λs)
    w = zero(eltype(λs))
    for i ∈ eachindex(λs)
        new_w = get_w(λs, i)
        if real(λs[i]) - new_w < 0
            break
        else
            w = new_w
        end
    end

    for (n, λ) in enumerate(λs)
        λs[n] = max(λ - w, zero(w))
    end

end

"""
    project2density(ρ)

Project a Hermitian matrix `ρ` to a density matrix by setting the negative eigenvalues to zero and normalizing the trace to 1.
"""
function project2density!(ρ)
    vals, vecs = eigen!(Hermitian(ρ), sortby=x -> -x)
    project_onto_simplex!(vals)
    broadcast!(√, vals, vals)

    rmul!(vecs, Diagonal(vals))
    mul!(ρ, vecs, vecs')
end

function project2density(ρ)
    σ = copy(ρ)
    project2density!(σ)
    σ
end

"""
    project2pure(ρ)

Project a Hermitian matrix `ρ` to a pure state by returning the eigenvector corresponding to the largest eigenvalue.
"""
function project2pure(ρ)
    F = eigen(Hermitian(ρ))
    F.vectors[:, end] # the last eigenvector is the one with the largest eigenvalue
end

"""
    isposdef!(ρ, xs, set)

Calculate the linear combination of the elements of `set` with the coefficients `xs` and check if the result is a positive definite matrix.
"""
function isposdef!(ρ, θ)
    density_matrix_reconstruction!(ρ, θ)
    isposdef!(ρ)
end

"""
    get_projector(ψ)

Calculate the matrix representing the projection operator over the state `ψ`.
"""
get_projector(ψ) = ψ * ψ'

"""
    polarization_state(::Val{S}, ::Type{T}=ComplexF32) where {S, T}

Return the polarization state corresponding to the symbol `S` as a vector of type `T`.
`S` can be one of the following symbols:
- `:H` for horizontal polarization
- `:V` for vertical polarization
- `:D` for diagonal polarization
- `:A` for antidiagonal polarization
- `:R` for right-handed circular polarization
- `:L` for left-handed circular polarization
"""
polarization_state(::Val{:H}, ::Type{T}=ComplexF32) where {T} = T[1, 0]
polarization_state(::Val{:V}, ::Type{T}=ComplexF32) where {T} = T[0, 1]
polarization_state(::Val{:D}, ::Type{T}=ComplexF32) where {T} = T[1/√2, 1/√2]
polarization_state(::Val{:A}, ::Type{T}=ComplexF32) where {T} = T[1/√2, -1/√2]
polarization_state(::Val{:R}, ::Type{T}=ComplexF32) where {T} = T[1/√2, -im/√2]
polarization_state(::Val{:L}, ::Type{T}=ComplexF32) where {T} = T[1/√2, im/√2]