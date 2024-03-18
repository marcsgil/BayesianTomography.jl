function log_likelihood(outcomes, operators, xs, ancilla)
    mul!(ancilla, operators, xs)
    map!(log, ancilla, ancilla)
    outcomes ⋅ ancilla
end

function proposal!(x, x₀, τ)
    x[begin] = zero(eltype(x))
    _x = @view x[begin+1:end]
    randn!(_x,)
    map!(x -> x * τ, _x, _x)
    map!(+, x, x, x₀)
end

function acceptance!(x₀, x, y₀, f)
    y = f(x)
    if y₀ - y ≤ log(rand())
        @. x₀ = x
        return y, true
    else
        return y₀, false
    end
end

function metropolisHastings(f, x₀, nsamples, nwarm, τ; verbose=false)
    out_of_domain = 0
    accepted = 0
    L = length(x₀)
    d = Int(√L)
    basis = gell_man_matrices(d)

    y₀ = f(x₀)
    x = similar(x₀)

    T = eltype(x₀)
    ρ = Matrix{complex(T)}(undef, d, d)

    for _ ∈ 1:nwarm
        proposal!(x, x₀, τ)
        if isposdef!(ρ, x, basis)
            y₀, _ = acceptance!(x₀, x, y₀, f)
        end
    end

    stat = CovMatrix(T, L)
    for _ ∈ 1:nsamples
        proposal!(x, x₀, τ)
        if isposdef!(ρ, x, basis)
            y₀, is_accepted = acceptance!(x₀, x, y₀, f)
            accepted += is_accepted
            fit!(stat, x₀)
        else
            out_of_domain += 1
        end
    end

    if verbose
        println("Out of domain rate: ", out_of_domain / nsamples)
        println("Acceptance rate: ", accepted / nsamples)
    end

    stat
end

struct BayesianInference{T<:Real}
    povm::Matrix{T}
    nsamples::Int
    nwarm::Int
    τ::T
    function BayesianInference(povm::AbstractArray{Matrix{T2}}, nsamples, nwarm, τ=1.0f-3) where {T2}
        T1 = real(T2)
        basis = gell_man_matrices(size(first(povm), 1))
        f(F) = real_orthogonal_projection(F, basis)
        new{T1}(stack(f, povm, dims=1), nsamples, nwarm, τ)
    end
end

function reduced_representation(povm, outcomes)
    reduced_outcomes = reduced_representation(outcomes)
    reduced_povm = similar(povm, size(reduced_outcomes, 2), size(povm, 2))

    for n ∈ axes(reduced_povm, 2), m ∈ axes(reduced_povm, 1)
        reduced_povm[m, n] = povm[Int(reduced_outcomes[1, m]), n]
    end

    T = eltype(povm)
    reduced_povm, map(T, view(reduced_outcomes, 2, :))
end


function prediction(outcomes, method::BayesianInference{T}; nchains=1, verbose=false) where {T}
    reduced_povm, reduced_outcomes = reduced_representation(method.povm, outcomes)

    d = Int(√size(reduced_povm, 2))
    x₀ = zeros(T, d^2)
    x₀[begin] = 1 / √d

    stats = fill(CovMatrix(T, d^2), nchains)

    Threads.@threads for n ∈ eachindex(stats)
        ancilla = similar(reduced_outcomes, float(eltype(reduced_outcomes)))
        posterior(x) = log_likelihood(reduced_outcomes, reduced_povm, x, ancilla)
        stats[n] = metropolisHastings(posterior, x₀, method.nsamples ÷ nchains, method.nwarm, method.τ, verbose=n == 1 && verbose)
    end

    for n ∈ 2:nchains
        merge!(stats[1], stats[n])
    end

    return linear_combination(mean(stats[1]), gell_man_matrices(d)), cov(stats[1])
end