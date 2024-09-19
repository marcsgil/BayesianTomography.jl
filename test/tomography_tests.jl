symbols = [:H, :V, :D, :A, :R, :L]
measurements = [get_projector(polarization_state(Val(s))) for s in symbols]

problem = StateTomographyProblem(measurements)
li = LinearInversion(problem)
bi = BayesianInference(problem)
methods = [LinearInversion(problem), MaximumLikelihood(problem), BayesianInference(problem)]
names = ["Linear Inversion", "Maximum Likelihood", "Bayesian Inference"]
Ns = [10^6, 10^4, 10^4]

function test_method(method, N, ρ::AbstractMatrix)
    outcomes = simulate_outcomes(ρ, measurements, N)
    σ = prediction(outcomes, method)[1]
    @test fidelity(ρ, σ) ≥ 0.99
    @test isposdef(σ)
end

function test_method(method, N, ψ::AbstractVector)
    outcomes = simulate_outcomes(ψ, measurements, N)
    σ = prediction(outcomes, method)[1]
    φ = project2pure(σ)
    @test fidelity(ψ, φ) ≥ 0.99
end

for (method, N, name) ∈ zip(methods, Ns, names)
    @testset "$name Tomography" begin
        for ψ ∈ eachslice(sample(HaarVector(2), 10), dims=2)
            test_method(method, N, ψ)
        end

        for ρ ∈ eachslice(sample(GinibreEnsamble(2), 10), dims=3)
            test_method(method, N, ρ)
        end
    end
end