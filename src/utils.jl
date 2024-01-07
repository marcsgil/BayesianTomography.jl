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

function simulate_outcomes(ψ::AbstractArray, operators, N, atol=1e-3)
    probs = [real(dot(ψ, E, ψ)) for E in operators]
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

function array_representation(operators, outcomes::Dict{T1,T2}) where {T1,T2}
    result = similar(operators, T2)
    for n ∈ eachindex(result)
        result[n] = get(outcomes, n, 0)
    end
    result
end