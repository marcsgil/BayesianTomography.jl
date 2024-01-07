module BayesianTomography

using Distributions, LinearAlgebra
using LogDensityProblems, AdvancedMH, MCMCChains, AbstractMCMC
using LogDensityProblemsAD, ForwardDiff, Random
using AdvancedHMC
using Logging
using Optimization, OptimizationOptimJL, OptimizationOptimisers, Optim, LineSearches
using Parameters, UnPack

include("utils.jl")
export compose_povm, unitary_transform, augment_povm, simulate_outcomes, circular_mean

using ClassicalOrthogonalPolynomials, Integrals
include("position_operators.jl")
export assemble_position_operators, hg_product, hg, hermite_position_operator
export MaximumLikelihood, MetropolisHastings, HamiltonianMC

include("samplers.jl")
export sample_haar_unitary, sample_haar_vector

include("pure_states.jl")
export hurwitz_parametrization, f, log_likellyhood, log_prior,
    PureLogPosterior, sample_posterior, prediction, random_angles, hg

end