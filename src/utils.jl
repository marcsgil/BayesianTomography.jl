"""
    simulate_outcomes(ψ::AbstractVector, povm, N; atol=1e-3)
    simulate_outcomes(ρ::AbstractMatrix, povm, N; atol=1e-3)
    simulate_outcomes(probs, N; atol=1e-3)

Simulate the `N` outcomes of a quantum measurement represented by a `povm` on a quantum state.

The state can be pure or mixed, and it is represented by a vector `ψ` or a density matrix `ρ`, respectively.
Alternativelly, one can directly provide the probabilities of the outcomes in the `probs` array.

`atol` is the absolute tolerance for the probabilities to be considered non-negative and to sum to 1.
"""
function simulate_outcomes(ψ::AbstractVector, povm, N; atol=1e-3)
    probs = [real(dot(ψ, E, ψ)) for E in povm]
    simulate_outcomes(probs, N; atol)
end

function simulate_outcomes(ρ::AbstractMatrix, povm, N; atol=1e-3)
    probs = [real(ρ ⋅ E) for E in povm]
    simulate_outcomes(probs, N; atol)
end

function simulate_outcomes(probs, N; atol=1e-3)
    @assert minimum(probs) ≥ -atol "The probabilities must be non-negative"
    S = sum(probs)
    @assert isapprox(S, 1; atol) "The sum of the probabilities is not 1, but $S"
    dist = Categorical(map(x -> x > 0 ? x : zero(x), normalize(vec(probs), 1)))

    complete_representation(History(rand(dist, N)), length(probs))
end


"""
    simulate_outcomes!(probs, N; atol=1e-3)

Simulate the `N` outcomes of a probability specified by the `probs` array.
The results are stored in the `probs` array.

`atol` is the absolute tolerance for the probabilities to be considered non-negative and to sum to 1.
"""
function simulate_outcomes!(probs, N; atol=1e-3)
    @assert minimum(probs) ≥ -atol "The probabilities must be non-negative"
    S = sum(probs)
    @assert isapprox(S, 1; atol) "The sum of the probabilities is not 1, but $S"

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
    sqrt_ρ = sqrt(ρ)
    abs2(tr(sqrt(sqrt_ρ * σ * sqrt_ρ)))
end

fidelity(ψ::AbstractVector, φ::AbstractVector) = abs2(ψ ⋅ φ)

"""
    project2density(ρ)

Project a Hermitian matrix `ρ` to a density matrix by setting the negative eigenvalues to zero and normalizing the trace to 1.
"""
function project2density(ρ)
    F = eigen(hermitianpart(ρ))
    λs = [λ > 0 ? λ : 0 for λ ∈ real.(F.values)]
    normalize!(λs, 1)
    sum(λ * v * v' for (λ, v) ∈ zip(λs, eachcol(F.vectors)))
end

"""
    project2pure(ρ)

Project a Hermitian matrix `ρ` to a pure state by returning the eigenvector corresponding to the largest eigenvalue.
"""
function project2pure(ρ)
    F = eigen(hermitianpart(ρ))
    F.vectors[:, end] # the last eigenvector is the one with the largest eigenvalue
end

"""
    orthogonal_projection(ρ, set)

Calculate the orthogonal projection of `ρ` onto `set`.

`set` is an array with one more dimension than `ρ`.
"""
function orthogonal_projection(ρ, set)
    @assert ndims(ρ) + 1 == ndims(set) """
    \nndims(ρ) + 1 != ndims(set) 
    Got ndims(ρ) = $(ndims(ρ)) and ndims(set) = $(ndims(set))
    """
    [ρ ⋅ Ω / (Ω ⋅ Ω) for Ω ∈ eachslice(set, dims=ndims(set))]
end

"""
    real_orthogonal_projection(ρ, set)

Calculate the real part of the orthogonal projection of `ρ` onto `set`.

`set` is an array with one more dimension than `ρ`.

This function is useful when the projection is expected to be real, but numerical errors may introduce small imaginary parts.
"""
function real_orthogonal_projection(ρ, set)
    @assert ndims(ρ) + 1 == ndims(set) """
    \nndims(ρ) + 1 != ndims(set) 
    Got ndims(ρ) = $(ndims(ρ)) and ndims(set) = $(ndims(set))
    """
    [real(ρ ⋅ Ω / (Ω ⋅ Ω)) for Ω ∈ eachslice(set, dims=ndims(set))]
end

"""
    linear_combination(xs, set)

Calculate the linear combination of the elements of `set` with the coefficients `xs`.
"""
function linear_combination(xs, set)
    sum(x * Ω for (x, Ω) ∈ zip(xs, eachslice(set, dims=ndims(set))))
end

"""
    linear_combination!(ρ, xs, set)

Calculate the linear combination of the elements of `set` with the coefficients `xs` and store the result in `ρ`.
"""
function linear_combination!(ρ, xs, set)
    fill!(ρ, zero(eltype(set)))
    for (x, Ω) in zip(xs, eachslice(set, dims=ndims(set)))
        @. ρ += x * Ω
    end
end

"""
    isposdef!(ρ, xs, set)

Calculate the linear combination of the elements of `set` with the coefficients `xs` and check if the result is a positive definite matrix.
"""
function isposdef!(ρ, xs, set)
    linear_combination!(ρ, xs, set)
    isposdef!(ρ)
end

"""
    cond(povm::Union{AbstractArray{T},AbstractMatrix{T}}, p::Real=2) where {T<:AbstractMatrix}

Calculate the condition number of the linear transformation associated with the `povm`.
"""
function cond(povm::Union{AbstractArray{T},AbstractMatrix{T}}, p::Real=2) where {T<:AbstractMatrix}
    d = size(first(povm), 1)
    set = gell_mann_matrices(d)
    A = stack(F -> real_orthogonal_projection(F, set), povm, dims=1)
    cond(A, p)
end

"""
    maximally_mixed_state(d, ::Type{T}) where {T}

Returns the maximally mixed state of dimension `d`, represented as a vector of projections in the generalized Gell-Mann basis.

The maximally mixed state is defined as `ρ = I / d`.

Se also [`gell_mann_matrices`](@ref).
"""
function maximally_mixed_state(d, ::Type{T}) where {T}
    x = zeros(T, d^2)
    x[begin] = 1 / √d
    x
end

function fisher!(F, T::AbstractArray, probs)
    I = findall(x -> x > 0, vec(probs))
    @tullio F[i, j] = T[I[k], i] * T[I[k], j] / probs[I[k]]
end

function fisher!(F, mthd, θs)
    probs = get_probs(mthd, θs)
    fisher!(F, (@view mthd.povm[:, begin+1:end]), probs)
end

function fisher(mthd, θs)
    D = mthd.dim^2 - 1
    F = Matrix{eltype(θs)}(undef, D, D)
    fisher!(F, mthd, θs)
    F
end