module BayesianTomography

using Distributions, Integrals, OnlineStats, Tullio, LinearAlgebra, Random
import LinearAlgebra: isposdef!, isposdef

include("hermitian_basis.jl")
export gell_man_matrices, basis_decomposition, Z_matrix, X_matrix, Y_matrix, triangular_indices

include("bayesian_inference.jl")
include("linear_inversion.jl")
export BayesianInference, LinearInversion, prediction

include("augmentation.jl")
export compose_povm, unitary_transform!, augment_povm

include("representations.jl")
export dict2array, array2dict, history2array, history2dict

include("samplers.jl")
export sample_haar_unitary, sample_haar_vector

include("utils.jl")
export simulate_outcomes, simulate_outcomes!, fidelity, project2density, project2pure
export linear_combination, linear_combination!, isposdef!

include("position_operators.jl")
export assemble_position_operators


end