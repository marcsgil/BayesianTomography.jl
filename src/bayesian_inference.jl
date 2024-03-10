function log_likelihood(outcomes, operators, xs, ancilla)
    mul!(ancilla, operators, xs)
    map!(log, ancilla, ancilla)
    outcomes ⋅ ancilla
end

function update_x!(x, x₀, β, L)
    @. x = x₀
    for n ∈ 2:L
        x[n] += randn() * β
    end
end

function update_x₀!(x₀, x, y₀, f, dist)
    y = f(x)
    if y₀ - y ≤ rand(dist)
        @. x₀ = x
        return y
    else
        return y₀
    end
end

function update_x₀!(x₀, x, y₀, f, dist, stats)
    y = f(x)
    if y₀ - y ≤ rand(dist)
        @. x₀ = x
        fit!(stats, Vector(x₀))
        return y
    else
        return y₀
    end
end

function metropolisHastings(f, x₀, nsamples, nwarm=0; β=1e-3)
    dist = Exponential()
    L = length(x₀)
    d = Int(√L)
    basis = get_hermitian_basis(d)

    y₀ = f(x₀)
    x = similar(x₀)

    T = eltype(x₀)
    ρ = Matrix{complex(T)}(undef, d, d)

    for _ ∈ 1:nwarm
        update_x!(x, x₀, β, L)

        if isposdef!(ρ, x, basis)
            y₀ = update_x₀!(x₀, x, y₀, f, dist)
        end
    end

    stat = CovMatrix(T, L)
    for _ ∈ 1:nsamples
        update_x!(x, x₀, β, L)

        if isposdef!(ρ, x, basis)
            y₀ = update_x₀!(x₀, x, y₀, f, dist, stat)
        end
    end

    stat
end

struct BayesianInference{T<:Real,N}
    povm::Array{Vector{T},N}
    nsamples::Int
    nwarm::Int
    function BayesianInference(povm::AbstractArray{Matrix{T2},N}, nsamples, nwarm) where {T2,N}
        T1 = real(T2)
        d = size(first(povm), 1)
        basis = get_hermitian_basis(d)
        new{T1,N}([real_representation(Ω, basis) for Ω ∈ povm], nsamples, nwarm)
    end
end

function efective_povm(povm, observations)
    new_povm = Matrix{eltype(povm)}(undef, length(observations), size(povm, 2))
    new_obs = Vector{Int}(undef, length(observations))

    for (n, pair) ∈ enumerate(observations)
        new_povm[n, :] = povm[pair.first, :]
        new_obs[n] = pair.second
    end

    new_povm, new_obs
end

function prediction(outcomes, method::BayesianInference)
    povm, flat_outcomes = efective_povm(method.povm |> vec |> stack |> transpose, outcomes)

    d = Int(√length(first(method.povm)))
    x₀ = vcat(1 / √d, zeros(d^2 - 1))

    nt = Threads.nthreads()
    stats = fill(CovMatrix(eltype(x₀), length(x₀)), nt)

    Threads.@threads for n ∈ eachindex(stats)
        ancilla = similar(flat_outcomes, Float64)
        posterior(x) = log_likelihood(flat_outcomes, povm, x, ancilla)
        stats[n] = metropolisHastings(posterior, x₀, method.nsamples ÷ nt, method.nwarm)
    end

    for n ∈ 2:nt
        merge!(stats[1], stats[n])
    end

    return stats[1]
end