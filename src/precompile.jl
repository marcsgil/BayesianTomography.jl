@setup_workload begin
    bs_povm = [[1.0 0im; 0 0], [0 0; 0 1]]
    half_wave_plate = [1 1; 1 -1] / √2
    quarter_wave_plate = [1 im; im 1] / √2

    @compile_workload begin
        povm = augment_povm(bs_povm, half_wave_plate, quarter_wave_plate, weights=[1 / 2, 1 / 4, 1 / 4])
        problem = StateTomographyProblem(povm)

        li = LinearInversion(problem)
        #bi = BayesianInference(povm)

        ρ = sample(GinibreEnsamble(2))

        outcomes = simulate_outcomes(ρ, povm, 1)
        σ, _ = prediction(outcomes, li)
        fidelity(ρ, σ)

        outcomes = simulate_outcomes(ρ, povm, 1)
        #σ, _ = prediction(outcomes, bi, nsamples=1, nwarm=1)

        ψ = sample(HaarVector(2))

        outcomes = simulate_outcomes(ψ, povm, 1)
        φ, _ = prediction(outcomes, li)
        φ = project2pure(φ)
        fidelity(ψ, φ)

        outcomes = simulate_outcomes(ψ, povm, 1)
        #φ, _ = prediction(outcomes, bi, nsamples=1, nwarm=1)
    end
end