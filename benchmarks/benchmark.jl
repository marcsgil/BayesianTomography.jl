using BayesianTomography
bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
quater_wave_plate = [1 im; im 1] / √2

povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, probabilities=[1 / 2, 1 / 4, 1 / 4])
li = LinearInversion(povm)
bi = BayesianInference(povm)

ρ = sample(GinibreEnsamble(2))

outcomes = simulate_outcomes(ρ, povm, 10^6)
σ = prediction(outcomes, li)
@b prediction($outcomes, $li)

outcomes = simulate_outcomes(ρ, povm, 10^4)
σ, _ = prediction(outcomes, bi);
@benchmark prediction($outcomes, $bi)
