function log_and_gradient!(outcomes, operators, xs, ∇ℓπ, cache1, cache2)
    mul!(cache1, operators, xs)
    map!(/, cache2, operators, cache1)
    map!(log, cache1, cache1)
    mul!(∇ℓπ, operators', cache2)
    outcomes ⋅ cache1
end

function prepare_distribution!(μ, Σ, x, ∇ℓπ, σ, A)
    BLAS.gemv!('N', σ^2 / 2, A, ∇ℓπ, false, μ)
    μ .+= x
    @. Σ = σ^2 * A
    MvNormal((@view μ[begin+1:end]), @view(Σ[begin+1:end, begin+1:end]))
end

function proposal!(x, dist)
    _x = @view x[begin+1:end]
    rand!(dist, _x)
end

function h(x, x₀, ∇ℓπ, σ, A)
    _∇ℓπ = @view ∇ℓπ[begin+1:end]
    _A = @view A[begin+1:end, begin+1:end]
    (x ⋅ ∇ℓπ - x₀ ⋅ ∇ℓπ - σ^2 * dot(_∇ℓπ, _A, _∇ℓπ) / 4) / 2
end

function proposal_ratio(x, x₀, ∇ℓπ, ∇ℓπ₀, σ, A)
    h(x, x₀, ∇ℓπ₀, σ, A) - h(x₀, x, ∇ℓπ, σ, A)
end

function acceptance!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ, A)
    ℓπ = f(x, ∇ℓπ)
    if ℓπ₀ - ℓπ + proposal_ratio(x, x₀, ∇ℓπ, ∇ℓπ₀, σ, A) ≤ rand(Exponential())
        @. x₀ = x
        @. ∇ℓπ₀ = ∇ℓπ
        return ℓπ, true
    else
        return ℓπ₀, false
    end
end

function step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ, A, μ, Σ, ρ, basis, stats)
    is_in_domain = false
    not_in_domain_count = -1
    T = typeof(σ)

    while !is_in_domain
        dist = prepare_distribution!(μ, Σ, x₀, ∇ℓπ₀, σ, A)
        proposal!(x, dist)

        is_in_domain = isposdef!(ρ, x, basis)
        not_in_domain_count += 1

        if !is_in_domain && not_in_domain_count > 100
            σ *= T(0.99)
        end
    end

    ℓπ₀, is_accepted = acceptance!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ, A)
    fit!(stats, x₀)

    ℓπ₀, is_accepted, not_in_domain_count, σ
end

function update_step_size(σ, acceptance_ratio, target, min, max)
    T = typeof(σ)
    if target < acceptance_ratio && σ < max
        return σ * T(1.01)
    else
        if σ > min
            return σ * T(0.99)
        else
            return σ
        end
    end
end

function sample_markov_chain(f, x₀, nsamples, nwarm; verbose=false, σ=oftype(eltype(x₀), 1e-2))
    L = length(x₀)
    d = Int(√L)

    target = 0.574
    minimum = 1e-8
    maximum = 100

    x = copy(x₀)
    ∇ℓπ₀ = similar(x)
    ∇ℓπ = similar(x)
    ℓπ₀ = f(x₀, ∇ℓπ₀)

    T = eltype(x₀)
    ρ = Matrix{complex(T)}(undef, d, d)
    basis = gell_man_matrices(d)

    μ = similar(x)
    A = Matrix{T}(I, L, L)
    Σ = similar(A)

    global_out_of_domain_count = 0
    global_accepted_count = 0

    stats = CovMatrix(T, L)
    for n ∈ 1:nwarm
        ℓπ₀, is_accepted, not_in_domain_count, σ = step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ, A, μ, Σ, ρ, basis, stats)
        global_out_of_domain_count += not_in_domain_count
        global_accepted_count += is_accepted
        σ = update_step_size(σ, global_accepted_count / n, target, minimum, maximum)
    end

    """copy!(A, cov(stats))
    A *= d / tr(A)"""

    global_out_of_domain_count = 0
    global_accepted_count = 0

    stat = CovMatrix(T, L)
    for n ∈ 1:nsamples
        """if n % 100 == 0
            copy!(A, cov(stats))
            A *= d / tr(A)
        end"""
        ℓπ₀, is_accepted, not_in_domain_count, σ = step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ, A, μ, Σ, ρ, basis, stats)
        global_out_of_domain_count += not_in_domain_count
        global_accepted_count += is_accepted
        σ = update_step_size(σ, global_accepted_count / n, target, minimum, maximum)
        @show x₀
        @show ℓπ₀
    end

    if verbose
        @info "Run information:"
        println("Out of domain rate: ", global_out_of_domain_count / nsamples)
        println("Acceptance rate: ", global_accepted_count / nsamples)
        println("σ: ", σ)
    end

    stat
end

struct BayesianInference{T<:Real}
    povm::Matrix{T}
    nsamples::Int
    nwarm::Int
    function BayesianInference(povm::AbstractArray{Matrix{T}}, nsamples, nwarm) where {T}
        basis = gell_man_matrices(size(first(povm), 1))
        f(F) = real_orthogonal_projection(F, basis)
        new{real(T)}(stack(f, povm, dims=1), nsamples, nwarm)
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


function prediction(outcomes, method::BayesianInference{T}; verbose=false, σ=T(1e-5)) where {T}
    reduced_povm, reduced_outcomes = reduced_representation(method.povm, outcomes)

    d = Int(√size(reduced_povm, 2))
    x₀ = zeros(T, d^2)
    x₀[begin] = 1 / √d

    cache1 = similar(reduced_outcomes, float(eltype(reduced_outcomes)))
    cache2 = similar(cache1)
    posterior(x, ∇ℓπ) = log_and_gradient!(reduced_outcomes, reduced_povm, x, ∇ℓπ, cache1, cache2)
    stats = sample_markov_chain(posterior, x₀, method.nsamples, method.nwarm; verbose, σ)

    return linear_combination(mean(stats), gell_man_matrices(d)), cov(stats)
end