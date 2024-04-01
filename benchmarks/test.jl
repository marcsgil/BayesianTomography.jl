using BayesianTomography, FiniteDifferences, LinearAlgebra

function BayesianTomography.fidelity(xs::AbstractVector, ρ::AbstractMatrix, method)
    σ = linear_combination(xs, method.basis)
    fidelity(ρ, σ)
end

function ∇fidelity(xs::AbstractVector, ρ::AbstractMatrix, method)
    f = x -> fidelity(x, ρ, method)
    grad(central_fdm(13, 1), f, xs)[1]
end

bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]] #POVM for a polarazing beam splitter
half_wave_plate = [1 1; 1 -1] / √2 #Unitary matrix for a half-wave plate
quarter_wave_plate = [1 im; im 1] / √2 #Unitary matrix for a quarter-wave plate

"""Augment the bs_povm with the action of half-wave plate and the quarter-wave plate. 
This is done because a single PBS is not enough to measure the polarization state of a photon."""
povm = augment_povm(bs_povm, half_wave_plate, quarter_wave_plate,
    weights=[1 / 2, 1 / 4, 1 / 4])

ρ = sample(ProductMeasure(2))
outcomes = simulate_outcomes(ρ, povm, 10^4)
mthd = BayesianInference(povm)
σ, xs, Σ = prediction(outcomes, mthd)
fidelity(ρ, σ)

fidelity(xs, ρ, mthd)
∇f = ∇fidelity(xs, ρ, mthd)
dot(∇f, Σ, ∇f)