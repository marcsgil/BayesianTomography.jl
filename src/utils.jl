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
    dist = Categorical(map(x -> x > 0 ? x : 0, normalize(vec(probs), 1)))

    complete_representation(History(rand(dist, N)), length(probs))
end

function simulate_outcomes!(probs, N; atol=1e-3)
    @assert minimum(probs) ≥ -atol "The probabilities must be non-negative"
    S = sum(probs)
    @assert isapprox(S, 1; atol) "The sum of the probabilities is not 1, but $S"

    dist = Categorical(map(x -> x > 0 ? x : 0, normalize(vec(probs), 1)))
    samples = rand(dist, N)

    Threads.@threads for n in eachindex(probs)
        probs[n] = count(x -> x == n, samples)
    end
end

function fidelity(ρ::AbstractMatrix, σ::AbstractMatrix)
    sqrt_ρ = sqrt(ρ)
    abs2(tr(sqrt(sqrt_ρ * σ * sqrt_ρ)))
end

fidelity(ψ::AbstractVector, φ::AbstractVector) = abs2(ψ ⋅ φ)

function project2density(ρ)
    F = eigen(hermitianpart(ρ))
    λs = [λ > 0 ? λ : 0 for λ ∈ real.(F.values)]
    normalize!(λs, 1)
    sum(λ * v * v' for (λ, v) ∈ zip(λs, eachcol(F.vectors)))
end

function project2pure(ρ)
    F = eigen(hermitianpart(ρ))
    F.vectors[:, end] # the last eigenvector is the one with the largest eigenvalue
end

function orthogonal_projection(ρ, basis)
    @assert ndims(ρ) + 1 == ndims(basis)
    [ρ ⋅ Ω / (Ω ⋅ Ω) for Ω ∈ eachslice(basis, dims=ndims(basis))]
end

function real_orthogonal_projection(ρ, basis)
    @assert ndims(ρ) + 1 == ndims(basis)
    [real(ρ ⋅ Ω / (Ω ⋅ Ω)) for Ω ∈ eachslice(basis, dims=ndims(basis))]
end

function linear_combination(xs, basis)
    ρ = Array{eltype(basis)}(undef, size(basis, 1), size(basis, 2))
    linear_combination!(ρ, xs, basis)
    ρ
end

function linear_combination!(ρ, xs, basis)
    fill!(ρ, zero(eltype(basis)))
    for (x, Ω) in zip(xs, eachslice(basis, dims=ndims(basis)))
        @. ρ += x * Ω
    end
end

function isposdef!(ρ, xs, basis)
    linear_combination!(ρ, xs, basis)
    isposdef!(ρ)
end

