module BayesianTomography

using Distributions, LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD, ForwardDiff, MCMCChains, AbstractMCMC
using AdvancedHMC, AdvancedMH
using Optim, LineSearches
using Parameters, UnPack

include("utils.jl")
export simulate_outcomes, array_representation, dict_representation, project2density

include("augmentation.jl")
export compose_povm, unitary_transform, augment_povm

using ClassicalOrthogonalPolynomials, Integrals
include("position_operators.jl")
export assemble_position_operators, hg_product, hg, hermite_position_operator

include("samplers.jl")
export sample_haar_unitary, sample_haar_vector

include("pure_states.jl")
export hurwitz_parametrization, f, log_likellyhood, log_prior,
    PureLogPosterior, sample_posterior, prediction, random_angles, hg
export MaximumLikelihood, MetropolisHastings, HamiltonianMC

include("linear_inversion.jl")
export LinearInversion, prediction

end