module BayesianTomography

using Distributions, OnlineStats, Tullio, LinearAlgebra, StatsBase, Random
import LinearAlgebra: isposdef!, isposdef, cond

include("gell_mann_matrices.jl")
export GellMannMatrices, gell_mann_projection!, gell_mann_reconstruction!, density_matrix_reconstruction!,
    gell_mann_projection, gell_mann_reconstruction, density_matrix_reconstruction

include("augmentation.jl")
export compose_povm, unitary_transform, unitary_transform!, augment_povm

include("representations.jl")
export complete_representation, reduced_representation, History

include("samplers.jl")
export sample, HaarUnitary, HaarVector, Simplex, ProductMeasure, GinibreEnsamble

include("utils.jl")
export simulate_outcomes, simulate_outcomes!, fidelity, project2density, project2density!, project2pure,
    isposdef!, maximally_mixed_state


include("tomography_problem.jl")
export StateTomographyProblem, cond, fisher, fisher!, get_probabilities, get_probabilities!

include("linear_inversion.jl")
include("bayesian_inference.jl")
include("max_likelihood.jl")
export BayesianInference, LinearInversion, MaximumLikelihood, prediction, get_probs, get_probs!

using PrecompileTools: @setup_workload, @compile_workload
include("precompile.jl")

end