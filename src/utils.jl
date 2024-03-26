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

function orthogonal_projection(ρ, set)
    @assert ndims(ρ) + 1 == ndims(set)
    [ρ ⋅ Ω / (Ω ⋅ Ω) for Ω ∈ eachslice(set, dims=ndims(set))]
end

function real_orthogonal_projection(ρ, set)
    @assert ndims(ρ) + 1 == ndims(set)
    [real(ρ ⋅ Ω / (Ω ⋅ Ω)) for Ω ∈ eachslice(set, dims=ndims(set))]
end

function linear_combination(xs, set)
    ρ = Array{eltype(set)}(undef, size(set, 1), size(set, 2))
    linear_combination!(ρ, xs, set)
    ρ
end

function linear_combination!(ρ, xs, set)
    fill!(ρ, zero(eltype(set)))
    for (x, Ω) in zip(xs, eachslice(set, dims=ndims(set)))
        @. ρ += x * Ω
    end
end

function isposdef!(ρ, xs, set)
    linear_combination!(ρ, xs, set)
    isposdef!(ρ)
end

