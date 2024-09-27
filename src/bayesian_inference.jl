"""
    log_likelihood!(∇ℓπ, buffer, outcomes, traceless_povm, correction, θ)

Returns the log-likelihood of the `outcomes` given the `traceless_povm`, `correction` and the state `θ`.
The gradient of the log-likelihood is stored in `∇ℓπ`.
`buffer` is an array used to store intermediate results.
"""
function log_likelihood!(∇ℓπ, buffer1, buffer2, outcomes, traceless_povm, correction, θ)
    get_probabilities!(buffer1, traceless_povm, correction, θ)
    broadcast!(/, buffer2, outcomes, buffer1)
    mul!(∇ℓπ, traceless_povm', buffer2)
    broadcast!(log, buffer1, buffer1)
    outcomes ⋅ buffer1
end

"""
    function proposal!(θ, θ₀, ∇ℓπ₀, σ)

Propose a new state `θ` given the current state `θ₀`.

The proposal is done by sampling a random vector `θ` from a normal distribution
with mean `θ₀ + σ^2 * ∇ℓπ₀ / 2` and covariance matrix `σ^2I`.
"""
function proposal!(θ, θ₀, ∇ℓπ₀, σ)
    randn!(θ)
    θ .*= σ
    @. θ += θ₀ + σ^2 * ∇ℓπ₀ / 2
end

function h(θ, θ₀, ∇ℓπ, σ)
    (θ ⋅ ∇ℓπ - θ₀ ⋅ ∇ℓπ - σ^2 * (∇ℓπ ⋅ ∇ℓπ) / 4) / 2
end

"""
    proposal_ratio(θ, θ₀, ∇ℓπ, ∇ℓπ₀, σ)

Returns the ratio of the transition probability of `θ₀` given `θ` and the `θ` given `θ₀`.

Used in the acceptance step of the MALA algorithm.
"""
function proposal_ratio(θ, θ₀, ∇ℓπ, ∇ℓπ₀, σ)
    h(θ, θ₀, ∇ℓπ, σ) - h(θ, θ₀, ∇ℓπ₀, σ)
end

"""
    acceptance!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, f, σ)

Accept or reject the proposed state `θ` given the current state `θ₀`.
If accepted, the state `θ₀` is updated to `θ` and the gradient `∇ℓπ₀` is updated to `∇ℓπ`.
Returns a tuple with the updated log-likelihood `ℓπ` and a boolean indicating if the state was accepted.
"""
function acceptance!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function!, σ)
    ℓπ = ℓπ_function!(∇ℓπ, θ)
    if ℓπ - ℓπ₀ + proposal_ratio(θ, θ₀, ∇ℓπ, ∇ℓπ₀, σ) ≥ log(rand())
        @. θ₀ = θ
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
    step!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function, parameters, ρ, basis, stats, n, target, min, max)

Perform a step of the MALA algorithm.
"""
function step!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function!, parameters, ρ, stats, n, target, min, max, chain)
    not_in_domain = true
    not_in_domain_count = -1

    # Keep proposing new states until a valid state is found
    while not_in_domain
        proposal!(θ, θ₀, ∇ℓπ₀, parameters[1])

        not_in_domain = !isposdef!(ρ, θ)
        not_in_domain_count += 1

        # Reduce σ if we keep getting out of domain states
        if not_in_domain && not_in_domain_count > 10
            parameters[1] *= 0.99
        end
    end

    # Accept or reject the proposed state
    ℓπ₀, is_accepted = acceptance!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function!, parameters[1])

    # Update the chain statistics
    fit!(stats, θ₀)

    if !isnothing(chain)
        chain[:, n] = θ₀
    end

    # Update the global statistics
    parameters[2] += is_accepted
    parameters[3] += not_in_domain_count

    # Update σ
    update_σ!(parameters, n, target, min, max)
    ℓπ₀
end

"""
    sample_markov_chain(ℓπ, θ₀::Vector{T}, nsamples, nwarm, basis;
        verbose=false,
        σ=oftype(T, 1e-2),
        target=0.574,
        minimum=1e-8,
        maximum=100) where {T<:Real}

Sample a Markov chain to sample the posterior of a quantum state tomography experiment using the MALA algorithm.
"""
function sample_markov_chain(ℓπ_function!, θ₀::Vector{T}, nsamples, nwarm;
    verbose=false,
    σ=oftype(T, 1e-2),
    target=0.574,
    min=1e-8,
    max=100,
    chain=nothing) where {T<:Real}

    L = length(θ₀)
    d = isqrt(L + 1)

    ρ = Matrix{complex(T)}(undef, d, d)
    @assert isposdef!(ρ, θ₀) "Initial state must be a valid density matrix. It must be positive semidefinite."

    θ = copy(θ₀)
    ∇ℓπ₀ = similar(θ)
    ∇ℓπ = similar(θ)
    ℓπ₀ = ℓπ_function!(∇ℓπ₀, θ₀)

    # σ, global_accepted_count, global_out_of_domain_count
    parameters = [σ, zero(T), zero(T)]
    stats = CovMatrix(T, L)
    for n ∈ 1:nwarm
        ℓπ₀ = step!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function!, parameters, ρ, stats, n, target, min, max, chain)
    end

    parameters[2] = zero(T)
    parameters[3] = zero(T)
    stats = CovMatrix(T, L)
    for n ∈ 1:nsamples
        ℓπ₀ = step!(θ₀, θ, ℓπ₀, ∇ℓπ₀, ∇ℓπ, ℓπ_function!, parameters, ρ, stats, n, target, min, max, chain)
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
struct BayesianInference end

"""
    prediction(outcomes, method::BayesianInference{T1, T2};
        verbose=false,
        σ=T1(1e-2),
        log_prior=θ -> zero(T1),
        θ₀=maximally_mixed_state(Int(√size(method.povm, 2)), T1),
        nsamples=10^4,
        nwarm=10^3,
        chain=nothing) where {T1}

Perform a Bayesian inference on the given `outcomes` using the [`BayesianInference`](@ref) `method`.

# Arguments

- `outcomes`: The outcomes of the experiment.
- `method::BayesianInference{T1, T2}`: The Bayesian inference method.
- `verbose=false`: Print information about the run.
- `σ=T(1e-2)`: The initial standard deviation of the proposal distribution.
- `log_prior=θ -> zero(T)`: The log-prior function.
- `θ₀=maximally_mixed_state(Int(√size(method.povm, 2)), T)`: The initial state of the chain.
- `nsamples=10^4`: The number of samples to take.
- `nwarm=10^3`: The number of warm-up samples to take.
- `chain=nothing`: If not `nothing`, store the chain in this matrix.

# Returns

A tuple with the mean state, its projection in `method.basis` and the covariance matrix.
The mean state is already returned in matrix form.
"""
function prediction(outcomes, measurement::Measurement{T1,T2}, ::BayesianInference;
    verbose=false,
    σ=T2(1e-2),
    log_prior=θ -> zero(T2),
    θ₀=zeros(T2, size(measurement.traceless_part, 2)),
    nsamples=10^4,
    nwarm=10^3,
    chain=nothing) where {T1,T2}

    vec_outcomes = vec(outcomes)
    I = findall(!iszero, vec_outcomes)
    _outcomes = vec_outcomes[I]
    traceless_part = measurement.traceless_part[I, :]
    trace_part = measurement.trace_part[I]
    buffer1 = similar(trace_part)
    buffer2 = similar(trace_part)

    ℓπ_function!(∇ℓπ, θ) = log_likelihood!(∇ℓπ, buffer1, buffer2, _outcomes, traceless_part, trace_part, θ) + log_prior(θ)
    stats = sample_markov_chain(ℓπ_function!, θ₀, nsamples, nwarm; verbose, σ, chain)

    θ = mean(stats)
    Σ = cov(stats)
    ρ = density_matrix_reconstruction(θ)

    ρ, θ, Σ
end