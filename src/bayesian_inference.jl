"""
    log_likelihood(outcomes, povm, x, ∇ℓπ, cache1, cache2)

Returns the log-likelihood of the `outcomes` given the `povm` and the state `x`.
The gradient of the log-likelihood is stored in `∇ℓπ`.
"""
function log_likelihood(outcomes, povm, x, ∇ℓπ, cache1, cache2)
    mul!(cache1, povm, x)
    map!(/, cache2, povm, cache1)
    map!(log, cache1, cache1)
    mul!(∇ℓπ, povm', cache2)
    outcomes ⋅ cache1
end

"""
    function proposal!(x, x₀, ∇ℓπ₀, σ)

Propose a new state `x` given the current state `x₀`.

The proposal is done by sampling a random vector `x` from a normal distribution
with mean `x₀ + σ^2 * ∇ℓπ₀ / 2` and covariance matrix `σ^2I`.
"""
function proposal!(x, x₀, ∇ℓπ₀, σ)
    _x = @view x[begin+1:end]
    _x₀ = @view x₀[begin+1:end]
    _∇ℓπ₀ = @view ∇ℓπ₀[begin+1:end]
    randn!(_x)
    _x .*= σ
    @. _x += _x₀ + σ^2 * _∇ℓπ₀ / 2
end

function h(x, x₀, ∇ℓπ, σ)
    _∇ℓπ = @view ∇ℓπ[begin+1:end]
    (x ⋅ ∇ℓπ - x₀ ⋅ ∇ℓπ - σ^2 * (_∇ℓπ ⋅ _∇ℓπ) / 4) / 2
end

"""
    proposal_ratio(x, x₀, ∇ℓπ, ∇ℓπ₀, σ)

Returns the ratio of the transition probability of `x₀` given `x` and the `x` given `x₀`.

Used in the acceptance step of the MALA algorithm.
"""
function proposal_ratio(x, x₀, ∇ℓπ, ∇ℓπ₀, σ)
    h(x₀, x, ∇ℓπ, σ) - h(x, x₀, ∇ℓπ₀, σ)
end

"""
    acceptance!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ)

Accept or reject the proposed state `x` given the current state `x₀`.
If accepted, the state `x₀` is updated to `x` and the gradient `∇ℓπ₀` is updated to `∇ℓπ`.
Returns a tuple with the updated log-likelihood `ℓπ` and a boolean indicating if the state was accepted.
"""
function acceptance!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function, σ)
    ℓπ = ℓπ_function(x, ∇ℓπ)
    if ℓπ - ℓπ₀ + proposal_ratio(x, x₀, ∇ℓπ, ∇ℓπ₀, σ) ≥ log(rand())
        @. x₀ = x
        @. ∇ℓπ₀ = ∇ℓπ
        return ℓπ, true
    else
        return ℓπ₀, false
    end
end

"""
    update_σ!(parameters, n, target, min, max)

Update the parameter `σ = parameters[1]` of the MALA algorithm given the current iteration `n` and the acceptance rate `parameters[2] / n`.
The target acceptance rate is `target` and the minimum and maximum values of `σ` are `min` and `max`, respectively.
"""
function update_σ!(parameters, n, target, min, max)
    if parameters[1] < min || parameters[2] / n > target
        parameters[1] *= 1.01
    end

    if parameters[1] > max || parameters[2] / n < target
        parameters[1] *= 0.99
    end
end


"""
    step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function, parameters, ρ, basis, stats, n, target, min, max)

Perform a step of the MALA algorithm.
"""
function step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function, parameters, ρ, basis, stats, n, target, min, max, chain)
    not_in_domain = true
    not_in_domain_count = -1

    # Keep proposing new states until a valid state is found
    while not_in_domain
        proposal!(x, x₀, ∇ℓπ₀, parameters[1])

        not_in_domain = !isposdef!(ρ, x, basis)
        not_in_domain_count += 1

        # Reduce σ if we keep getting out of domain states
        if not_in_domain && not_in_domain_count > 10
            parameters[1] *= 0.99
        end
    end

    # Accept or reject the proposed state
    ℓπ₀, is_accepted = acceptance!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function, parameters[1])

    # Update the chain statistics
    fit!(stats, x₀)

    if !isnothing(chain)
        chain[:, n] = x₀
    end

    # Update the global statistics
    parameters[2] += is_accepted
    parameters[3] += not_in_domain_count

    # Update σ
    update_σ!(parameters, n, target, min, max)
    ℓπ₀
end

"""
    sample_markov_chain(ℓπ, x₀::Vector{T}, nsamples, nwarm, basis;
        verbose=false,
        σ=oftype(T, 1e-2),
        target=0.574,
        minimum=1e-8,
        maximum=100) where {T<:Real}

Sample a Markov chain to sample the posterior of a quantum state tomography experiment using the MALA algorithm.
"""
function sample_markov_chain(ℓπ, x₀::Vector{T}, nsamples, nwarm, basis;
    verbose=false,
    σ=oftype(T, 1e-2),
    target=0.574,
    minimum=1e-8,
    maximum=100,
    chain=nothing) where {T<:Real}

    L = length(x₀)
    d = Int(√L)

    ρ = Matrix{complex(T)}(undef, d, d)


    @assert x₀[1] ≈ 1 / √d "Initial state must be a valid density matrix. The first element must be 1/√d."
    @assert isposdef!(ρ, x₀, basis) "Initial state must be a valid density matrix. It must be positive semidefinite."

    x = copy(x₀)
    ∇ℓπ₀ = similar(x)
    ∇ℓπ = similar(x)
    ℓπ₀ = ℓπ(x₀, ∇ℓπ₀)

    # σ, global_accepted_count, global_out_of_domain_count
    parameters = [σ, zero(T), zero(T)]
    stats = CovMatrix(T, L)
    for n ∈ 1:nwarm
        ℓπ₀ = step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ, parameters, ρ, basis, stats, n, target, minimum, maximum, nothing)
    end

    parameters[2] = zero(T)
    parameters[3] = zero(T)
    stats = CovMatrix(T, L)
    for n ∈ 1:nsamples
        ℓπ₀ = step!(x₀, x, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ, parameters, ρ, basis, stats, n, target, minimum, maximum, chain)
    end

    if verbose
        @info "Run information:"
        println("Final σ: ", parameters[1])
        println("Final acceptance rate: ", parameters[2] / nsamples)
        println("Final out of domain rate: ", parameters[3] / nsamples)
    end

    stats
end

"""
    BayesianInference(povm::AbstractArray{Matrix{T}},
        basis=gell_mann_matrices(size(first(povm), 1), complex(T))) where {T}

Create a Bayesian inference object from a POVM.

This is passed to the [`prediction`](@ref) method in order to perform the Bayesian inference.
"""
struct BayesianInference{T1<:Real,T2<:Union{Real,Complex}}
    povm::Matrix{T1}
    basis::Array{T2,3}
    function BayesianInference(povm::AbstractArray{Matrix{T}},
        basis=gell_mann_matrices(size(first(povm), 1), complex(T))) where {T}
        f(F) = real_orthogonal_projection(F, basis)
        new{real(T),complex(T)}(stack(f, povm, dims=1), basis)
    end
end


"""
    reduced_representation(povm, outcomes)

Returns a reduced representation of both the `povm` and the `outcomes`.

One determines the nonzero elements of `outcomes` and then selects the corresponding columns of the `povm`.

This function is used in the Bayesian inference to reduce the size of the problem by ignoring unobserved outcomes.
"""
function reduced_representation(povm, outcomes)
    reduced_outcomes = reduced_representation(outcomes)
    reduced_povm = similar(povm, size(reduced_outcomes, 2), size(povm, 2))

    for n ∈ axes(reduced_povm, 2), m ∈ axes(reduced_povm, 1)
        reduced_povm[m, n] = povm[Int(reduced_outcomes[1, m]), n]
    end

    T = eltype(povm)
    reduced_povm, map(T, view(reduced_outcomes, 2, :))
end

"""
    prediction(outcomes, method::BayesianInference{T};
        verbose=false,
        σ=T(1e-2),
        log_prior=x -> zero(T),
        x₀=maximally_mixed_state(Int(√size(method.povm, 2)), T),
        nsamples=10^4,
        nwarm=10^3,
        chain=nothing) where {T}

Perform a Bayesian inference on the given `outcomes` using the [`BayesianInference`](@ref) `method`.

# Arguments

- `outcomes`: The outcomes of the experiment.
- `method::BayesianInference{T}`: The Bayesian inference method.
- `verbose=false`: Print information about the run.
- `σ=T(1e-2)`: The initial standard deviation of the proposal distribution.
- `log_prior=x -> zero(T)`: The log-prior function.
- `x₀=maximally_mixed_state(Int(√size(method.povm, 2)), T)`: The initial state of the chain.
- `nsamples=10^4`: The number of samples to take.
- `nwarm=10^3`: The number of warm-up samples to take.
- `chain=nothing`: If not `nothing`, store the chain in this matrix.

# Returns

A tuple with the mean state and the covariance matrix.
The mean state is already returned in matrix form.
The covariance matrix is written in terms of the projections in the generalized Gell-Mann basis.
"""
function prediction(outcomes, method::BayesianInference{T};
    verbose=false,
    σ=T(1e-2),
    log_prior=x -> zero(T),
    x₀=maximally_mixed_state(Int(√size(method.povm, 2)), T),
    nsamples=10^4,
    nwarm=10^3,
    chain=nothing) where {T}

    reduced_povm, reduced_outcomes = reduced_representation(method.povm, outcomes)

    d = Int(√size(reduced_povm, 2))

    cache1 = similar(reduced_outcomes, float(eltype(reduced_outcomes)))
    cache2 = similar(cache1)
    posterior(x, ∇ℓπ) = log_likelihood(reduced_outcomes, reduced_povm, x, ∇ℓπ, cache1, cache2) + log_prior(x)
    stats = sample_markov_chain(posterior, x₀, nsamples, nwarm, method.basis; verbose, σ, chain)

    μ = mean(stats)
    Σ = cov(stats)
    linear_combination(μ, method.basis), μ, Σ
end