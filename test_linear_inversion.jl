using BayesianTomography, LinearAlgebra

pbs_operators = [[1 0; 0 0], [0 0; 0 1]]
half_wave = [1 1; 1 -1] / √2
quarter_wave = [1 im; im 1] / √2

polarization_Es = augment_povm(pbs_operators, half_wave, quarter_wave)
##
outcomes = simulate_outcomes(Matrix{Float64}(I, 2, 2) / 2, polarization_Es, 10^5)
probs = normalize(array_representation(outcomes, 6), 1)

LinearInversion(2)

prediction(probs, polarization_Es, LinearInversion(2))