function angular_transform!(angles)
    n = length(angles) ÷ 2
    @. angles[1:n] = acos(cos(@view angles[1:n]))
    @. angles[n+1:end] = mod2pi(@view angles[n+1:end])

    return nothing
end

function random_angles(d)
    θs = rand(Cosine(π / 2, π / 2), d ÷ 2)
    ϕs = rand(Uniform(0, 2π), d ÷ 2)
    vcat(θs, ϕs)
end

circular_mean(ϕs; dims=1:ndims(ϕs)) = mod2pi.(atan.(sum(sin, ϕs; dims), sum(cos, ϕs; dims)))

function simulate_outcomes(ψ::Vector{T}, operators, N; atol=1e-3) where {T}
    probs = [real(dot(ψ, E, ψ)) for E in operators]
    simulate_outcomes(probs, N; atol)
end

function simulate_outcomes(ρ::Matrix{T}, operators, N; atol=1e-3) where {T}
    probs = [real(ρ ⋅ E) for E in operators]
    simulate_outcomes(probs, N; atol)
end

function simulate_outcomes(probs, N; atol=1e-3)
    @assert minimum(probs) ≥ -atol "The probabilities must be non-negative"
    S = sum(probs)
    @assert isapprox(S, 1; atol) "The sum of the probabilities is not 1, but $S"
    dist = Categorical(map(x -> x > 0 ? x : 0, normalize(vec(probs), 1)))
    samples = rand(dist, N)

    outcomes = Dict{Int,Int}()
    for outcome ∈ samples
        outcomes[outcome] = get(outcomes, outcome, 0) + 1
    end
    outcomes
end



function project2density(ρ)
    F = eigen(hermitianpart(ρ))
    λs = [λ > 0 ? λ : 0 for λ ∈ real.(F.values)]
    normalize!(λs, 1)
    sum(λ * v * v' for (λ, v) ∈ zip(λs, eachcol(F.vectors)))
end