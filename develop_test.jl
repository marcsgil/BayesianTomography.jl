using BayesianTomography, LinearAlgebra

bs_povm = [[1.0 0im; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
quater_wave_plate = [1 im; im 1] / √2

povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, weights=[1 / 2, 1 / 4, 1 / 4])

povm[1]

problem = StateTomographyProblem(povm)

li = LinearInversion(problem)
#bi = BayesianInference(povm)

ρ = sample(GinibreEnsamble(2))

outcomes = simulate_outcomes(ρ, povm, 10^6)
θ = prediction(outcomes, li)

σ = similar(ρ)

density_matrix_reconstruction(θ) 

σ
ρ