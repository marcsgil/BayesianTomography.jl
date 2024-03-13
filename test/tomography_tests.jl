bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
quater_wave_plate = [1 im; im 1] / √2

povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, probabilities=[1 / 2, 1 / 4, 1 / 4])
li = LinearInversion(povm)
bi = BayesianInference(povm, 10^5, 10^3)

for _ ∈ 1:10
    #Mixed
    ρ = sample_ginibri_state(2)

    outcomes = simulate_outcomes(ρ, povm, 10^6)
    σ = prediction(outcomes, li)
    @test fidelity(ρ, σ) ≥ 0.99

    outcomes = simulate_outcomes(ρ, povm, 10^4)
    σ, _ = prediction(outcomes, bi)
    @test fidelity(ρ, σ) ≥ 0.99

    #Pure
    ψ = sample_haar_vector(2)

    outcomes = simulate_outcomes(ψ, povm, 10^6)
    φ = prediction(outcomes, li) |> project2pure
    @test fidelity(ψ, φ) ≥ 0.99

    outcomes = simulate_outcomes(ψ, povm, 10^4)
    φ, _ = prediction(outcomes, bi)
    φ = project2pure(φ)
    @test fidelity(ψ, φ) ≥ 0.99
end