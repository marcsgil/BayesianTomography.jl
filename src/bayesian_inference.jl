function log_likelihood(outcomes, operators, xs, cache)
    mul!(cache, operators, xs)
    map!(log, cache, cache)
    outcomes ⋅ cache
end

function log_and_gradient!(outcomes, operators, xs, grad, cache1, cache2)
    mul!(cache1, operators, xs)
    map!(/, cache2, operators, cache1)
    map!(log, cache1, cache1)
    mul!(grad, operators', cache2)
    outcomes ⋅ cache1
end

function h(z, v, grad, τ, A)
    _grad = @view grad[begin+1:end]
    (z ⋅ grad - v ⋅ grad - τ * dot(_grad, A, _grad) / 2) / 2
end

function proposal!(x, x₀, grad_x₀, τ, ρ, basis, Σ=I)
    is_in_domain = false
    not_in_domain_count = -1
    while !is_in_domain
        _x₀ = @view x₀[begin+1:end]
        _x = @view x[begin+1:end]
        _grad_x₀ = @view grad_x₀[begin+1:end]
        mul!(_x, Σ, _grad_x₀)
        _x .*= τ
        _x .+= _x₀
        Random.rand!(MvNormal(_x, 2 * τ * Σ), _x)
        is_in_domain = isposdef!(ρ, x, basis)
        not_in_domain_count += 1
        """if not_in_domain_count > 10^4
            τ /= 100
        end"""
    end
    return not_in_domain_count, τ
end

function acceptance!(x₀, x, y₀, grad₀, grad, f, τ, Σ=Matrix{eltype(x)}(I, length(x) - 1, length(x) - 1))
    y = f(x, grad)
    if y₀ - y + h(x, x₀, grad₀, τ, Σ) - h(x₀, x, grad, τ, Σ) ≤ rand(Exponential())
        @. x₀ = x
        @. grad₀ = grad
        return y, true
    else
        return y₀, false
    end
end

function update_step_size(τ, acceptance_ratio, not_in_domain_ratio, target, m, M)
    if target < acceptance_ratio && τ < M && not_in_domain_ratio < 10
        return τ * 1.01
    else
        if τ > m
            return τ * 0.99
        else
            return τ
        end
    end
    @show τ
end

function metropolisHastings(f, x₀, nsamples, nwarm, τ; verbose=false)
    L = length(x₀)
    d = Int(√L)
    basis = gell_man_matrices(d)
    target = 0.574
    minimum = 1e-8
    maximum = 1

    x = copy(x₀)
    grad₀ = similar(x)
    grad = similar(x)
    y₀ = f(x₀, grad₀)

    T = eltype(x₀)
    ρ = Matrix{complex(T)}(undef, d, d)

    out_of_domain = 0
    accepted = 0
    stat = CovMatrix(T, L)
    for n ∈ 1:nwarm
        not_in_domain_count, τ = proposal!(x, x₀, grad₀, τ, ρ, basis)
        out_of_domain += not_in_domain_count
        y₀, is_accepted = acceptance!(x₀, x, y₀, grad₀, grad, f, τ)
        accepted += is_accepted
        τ = update_step_size(τ, accepted / n, out_of_domain / n, target, minimum, maximum)
        fit!(stat, x₀)
    end


    Σ = @view cov(stat)[begin+1:end, begin+1:end]
    Σ *= d / tr(Σ)
    out_of_domain = 0
    accepted = 0
    stat = CovMatrix(T, L)
    for n ∈ 1:nsamples
        if n % 100 == 0
            Σ = @view cov(stat)[begin+1:end, begin+1:end]
            Σ *= d / tr(Σ)
        end
        not_in_domain_count, τ = proposal!(x, x₀, grad₀, τ, ρ, basis)
        out_of_domain += not_in_domain_count
        y₀, is_accepted = acceptance!(x₀, x, y₀, grad₀, grad, f, τ)
        accepted += is_accepted
        τ = update_step_size(τ, accepted / n, out_of_domain / n, target, minimum, maximum)
        fit!(stat, x₀)
    end

    if verbose
        println("Out of domain rate: ", out_of_domain / nsamples)
        println("Acceptance rate: ", accepted / nsamples)
        println("τ: ", τ)
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


function prediction(outcomes, method::BayesianInference{T}; verbose=false) where {T}
    reduced_povm, reduced_outcomes = reduced_representation(method.povm, outcomes)

    d = Int(√size(reduced_povm, 2))
    x₀ = zeros(T, d^2)
    x₀[begin] = 1 / √d

    stats = CovMatrix(T, d^2)
    cache1 = similar(reduced_outcomes, float(eltype(reduced_outcomes)))
    cache2 = similar(cache1)
    posterior(x, grad) = log_and_gradient!(reduced_outcomes, reduced_povm, x, grad, cache1, cache2)
    stats = metropolisHastings(posterior, x₀, method.nsamples, method.nwarm, method.τ; verbose)

    return linear_combination(mean(stats), gell_man_matrices(d)), cov(stats)
end