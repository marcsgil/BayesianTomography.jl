θ_transform(θ) = acos(cos(θ))
ϕ_transform(ϕ) = mod2pi(ϕ)

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

function compose_povm(args...)
    N = length(args)
    stack(arg / N for arg in args)
end

function unitary_transform(operators, unitary)
    [unitary' * operator * unitary for operator in operators]
end

function augment_povm(povm, unitaries...)
    compose_povm(povm, (unitary_transform(povm, unitary) for unitary ∈ unitaries)...)
end