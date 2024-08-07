module BayesianTomography

using Distributions, OnlineStats, Tullio, LinearAlgebra, StatsBase, Random
import LinearAlgebra: isposdef!, isposdef, cond

include("hermitian_basis.jl")
export gell_mann_matrices, basis_decomposition, Z_matrix, X_matrix, Y_matrix, triangular_indices

include("augmentation.jl")
export compose_povm, unitary_transform, unitary_transform!, augment_povm

include("representations.jl")
export complete_representation, reduced_representation, History

include("samplers.jl")
export sample, HaarUnitary, HaarVector, Simplex, ProductMeasure, GinibreEnsamble

include("utils.jl")
export simulate_outcomes, simulate_outcomes!, fidelity, project2density, project2pure,
    linear_combination, linear_combination!, isposdef!,
    real_orthogonal_projection, orthogonal_projection, cond, maximally_mixed_state,
    fisher_information

include("bayesian_inference.jl")
include("linear_inversion.jl")
export BayesianInference, LinearInversion, prediction

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]]
    half_wave_plate = [1 1; 1 -1] / √2
    quater_wave_plate = [1 im; im 1] / √2

    @compile_workload begin
        povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, weights=[1 / 2, 1 / 4, 1 / 4])
        li = LinearInversion(povm)
        bi = BayesianInference(povm)

        ρ = sample(GinibreEnsamble(2))

        outcomes = simulate_outcomes(ρ, povm, 1)
        σ, _ = prediction(outcomes, li)
        fidelity(ρ, σ)

        outcomes = simulate_outcomes(ρ, povm, 1)
        σ, _ = prediction(outcomes, bi, nsamples=1, nwarm=1)

        ψ = sample(HaarVector(2))

        outcomes = simulate_outcomes(ψ, povm, 1)
        φ, _ = prediction(outcomes, li)
        φ = project2pure(φ)
        fidelity(ψ, φ)

        outcomes = simulate_outcomes(ψ, povm, 1)
        φ, _ = prediction(outcomes, bi, nsamples=1, nwarm=1)
    end
end

end