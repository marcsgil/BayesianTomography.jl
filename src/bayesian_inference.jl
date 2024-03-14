function log_likelihood(outcomes, operators, xs, ancilla)
    mul!(ancilla, operators, xs)
    map!(log, ancilla, ancilla)
    outcomes ⋅ ancilla
end

function update_x!(x, x₀, σ)
    x[begin] = zero(eltype(x))
    _x = @view x[begin+1:end]
    randn!(_x,)
    map!(x -> x * σ, _x, _x)
    map!(+, x, x, x₀)
    nothing
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

function metropolisHastings(f, x₀, nsamples, nwarm=0; σ=1e-3)
    dist = Exponential()
    L = length(x₀)
    d = Int(√L)
    basis = gell_man_matrices(d)

    y₀ = f(x₀)
    x = similar(x₀)

    T = eltype(x₀)
    ρ = Matrix{complex(T)}(undef, d, d)

    for _ ∈ 1:nwarm
        update_x!(x, x₀, σ)

        if isposdef!(ρ, x, basis)
            y₀ = update_x₀!(x₀, x, y₀, f, dist)
        end
    end

    stat = CovMatrix(T, L)
    for _ ∈ 1:nsamples
        update_x!(x, x₀, σ)

        if isposdef!(ρ, x, basis)
            y₀ = update_x₀!(x₀, x, y₀, f, dist, stat)
        end
    end

    stat
end

struct BayesianInference{T<:Real,N}
    povm::Matrix{T}
    nsamples::Int
    nwarm::Int
    function BayesianInference(povm::AbstractArray{Matrix{T2},N}, nsamples, nwarm) where {T2,N}
        T1 = real(T2)
        basis = gell_man_matrices(size(first(povm), 1))
        f(F) = real_orthogonal_projection(F, basis)
        new{T1,N}(stack(f, povm, dims=1), nsamples, nwarm)
    end
end

function reduced_representation(povm, outcomes)
    reduced_outcomes = reduced_representation(outcomes)
    reduced_povm = similar(povm, size(reduced_outcomes, 2), size(povm, 2))

    for n ∈ axes(reduced_povm, 2), m ∈ axes(reduced_povm, 1)
        reduced_povm[m, n] = povm[Int(reduced_outcomes[1, m]), n]
    end

    reduced_povm, view(reduced_outcomes, 2, :)
end


function prediction(outcomes, method::BayesianInference)
    reduced_povm, reduced_outcomes = reduced_representation(method.povm, outcomes)

    d = Int(√size(reduced_povm, 2))
    x₀ = zeros(Float32, d^2)
    x₀[begin] = 1 / √d

    nt = Threads.nthreads()
    stats = fill(CovMatrix(eltype(x₀), length(x₀)), nt)

    Threads.@threads for n ∈ eachindex(stats)
        ancilla = similar(reduced_outcomes, float(eltype(reduced_outcomes)))
        posterior(x) = log_likelihood(reduced_outcomes, reduced_povm, x, ancilla)
        stats[n] = metropolisHastings(posterior, x₀, method.nsamples ÷ nt, method.nwarm)
    end

    for n ∈ 2:nt
        merge!(stats[1], stats[n])
    end

    return linear_combination(mean(stats[1]), gell_man_matrices(d)), cov(stats[1])
end