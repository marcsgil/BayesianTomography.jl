module BayesianTomography

using Distributions, LinearAlgebra, Tullio, LoopVectorization
using LogDensityProblems, LogDensityProblemsAD, ForwardDiff, MCMCChains, AbstractMCMC
using AdvancedHMC, AdvancedMH
using Optim, LineSearches
using Parameters, UnPack
using ClassicalOrthogonalPolynomials, Integrals, StaticArrays

include("utils.jl")
export simulate_outcomes, array_representation, dict_representation, project2density

include("augmentation.jl")
export compose_povm, unitary_transform!, augment_povm

include("position_operators.jl")
export assemble_position_operators

include("samplers.jl")
export sample_haar_unitary, sample_haar_vector

include("pure_states.jl")
export hurwitz_parametrization, f, g, log_likellyhood, log_prior,
    PureLogPosterior, sample_posterior, prediction, random_angles, hg
export MaximumLikelihood, MetropolisHastings, HamiltonianMC

include("linear_inversion.jl")
export LinearInversion, prediction

include("representations.jl")
export dict2array, array2dict, history2array, history2dict

end