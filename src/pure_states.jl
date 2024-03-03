function hurwitz_parametrization(angles)
    n = length(angles) ÷ 2
    T = complex(float(eltype(angles)))
    ψ = Vector{T}(undef, n + 1)
    θ = @view angles[1:n]
    ϕ = @view angles[n+1:2n]

    ψ[1] = 1

    for k ∈ 1:n
        ψ[k+1] = sin(θ[k] / 2) * ψ[k]
    end

    for k ∈ 1:n
        ψ[k] *= cos(θ[k] / 2)
    end

    for k ∈ 1:n
        ψ[k+1] *= cis(ϕ[k])
    end

    #SVector{n + 1}(ψ)
    ψ
end

function f(operator, ψ)
    log(real(dot(ψ, operator, ψ)))
end

function log_likellyhood(outcomes, operators, angles)
    ψ = hurwitz_parametrization(angles)
    sum(pair -> pair.second * f(operators[pair.first], ψ), pairs(outcomes))
end

function log_prior(angles)
    n = length(angles) ÷ 2
    θ = @view angles[1:n]
    sum(x -> log(cos(x[2] / 2)) + (2x[1] - 1) * log(sin(x[2] / 2)), enumerate(θ))
end

@with_kw struct MaximumLikelihood
    ntries::Int = 4
end

function prediction(outcomes, operators, method::MaximumLikelihood)
    f(x) = -log_likellyhood(outcomes, operators, x)
    n = size(first(operators), 1) - 1
    lower = zeros(2n)
    upper = vcat(fill(π, n), fill(2π, n))
    inner_optimizer = BFGS()

    sols = [
        optimize(f, lower, upper, random_angles(2n), Fminbox(inner_optimizer);
            autodiff=:forward) for i ∈ 1:method.ntries
    ]

    i = argmin(Optim.minimum(sol) for sol ∈ sols)
    Optim.minimizer(sols[i])
end

struct PureLogPosterior{T,N}
    outcomes::Dict{Int,Int}
    operators::Array{Matrix{T},N}
    dim::Int

    function PureLogPosterior(outcomes, operators)
        dim = size(first(operators), 1)
        T = eltype(first(operators))
        N = ndims(operators)
        new{T,N}(outcomes, operators, dim)
    end
end

function LogDensityProblems.capabilities(::Type{<:PureLogPosterior})
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(ℓ::PureLogPosterior) = 2 * (ℓ.dim - 1)

function LogDensityProblems.logdensity(ℓ::PureLogPosterior, angles)
    angular_transform!(angles)

    ll = log_likellyhood(ℓ.outcomes, ℓ.operators, angles)
    lp = log_prior(angles)

    ll + lp
end

abstract type BayesianMethod end

@with_kw struct MetropolisHastings <: BayesianMethod
    nsamples::Int = 2000
    nadapts::Int = 1000
    nchains::Int = 4
end

function sample_posterior(outcomes, operators, method::MetropolisHastings)
    @unpack nsamples, nadapts, nchains = method
    ℓ = PureLogPosterior(outcomes, operators)
    d = LogDensityProblems.dimension(ℓ)
    sampler = RWMH(MvNormal(zeros(d), I))
    initial_params = prediction(outcomes, operators, MaximumLikelihood())

    sample(
        ℓ,
        sampler,
        MCMCThreads(),
        nsamples,
        nchains;
        chain_type=Chains,
        discard_initial=nadapts,
        initial_params=fill(initial_params, nchains)
    )
end

@with_kw struct HamiltonianMC <: BayesianMethod
    nsamples::Int = 2000
    nadapts::Int = 1000
    nchains::Int = 4
    δ::AbstractFloat = 0.8
    α::AbstractFloat = 1.5
end

function sample_posterior(outcomes, operators, method::HamiltonianMC)
    @unpack nsamples, nadapts, nchains, δ, α = method
    ℓ = PureLogPosterior(outcomes, operators)
    d = LogDensityProblems.dimension(ℓ)

    metric = DiagEuclideanMetric(d)
    initial_params = prediction(outcomes, operators, MaximumLikelihood())
    hamiltonian = Hamiltonian(metric, ℓ, ForwardDiff)
    initial_ϵ = find_good_stepsize(hamiltonian, initial_params)
    integrator = TemperedLeapfrog(initial_ϵ, α)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(δ, integrator))
    sampler = HMCSampler(kernel, metric, adaptor)

    sample(
        ℓ,
        sampler,
        MCMCThreads(),
        nsamples,
        nchains;
        progress=false,
        chain_type=Chains,
        discard_initial=nadapts,
        initial_params=fill(initial_params, nchains)
    )
end

function prediction(outcomes, operators, method::BayesianMethod)
    chain = sample_posterior(outcomes, operators, method)
    n = size(first(operators), 1) - 1
    log_posteriors = chain[:lp]
    i = argmin(vec(mean(log_posteriors, dims=1)))
    θ = mean(x -> acos(cos(x)), chain.value[:, 1:n, i], dims=1)
    ϕ = circular_mean(Array(chain.value[:, n+1:2n, i]), dims=1)

    vcat(vec(θ), vec(ϕ))
end

