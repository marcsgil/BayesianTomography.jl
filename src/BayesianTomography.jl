module BayesianTomography

using Distributions, OnlineStats, Tullio, LinearAlgebra, Random
import LinearAlgebra: isposdef!, isposdef, cond
using Base.Threads: @spawn, nthreads
using Base.Iterators: partition

include("generalized_operators.jl")

include("gell_mann_matrices.jl")
export GellMannMatrices, gell_mann_projection!, gell_mann_reconstruction!, density_matrix_reconstruction!,
    gell_mann_projection, gell_mann_reconstruction, density_matrix_reconstruction,
    get_coefficients, reconstruction, get_coefficients!, reconstruction!

include("samplers.jl")
export sample, HaarUnitary, HaarVector, Simplex, ProductMeasure, GinibreEnsamble

include("utils.jl")
export simulate_outcomes, simulate_outcomes!, fidelity, project2density, project2density!, project2pure,
    isposdef!, maximally_mixed_state, get_projector, polarization_state


include("measurement.jl")
export Measurement, ProportionalMeasurement, cond, fisher, fisher!, get_probabilities, get_probabilities!

include("linear_inversion.jl")
include("max_likelihood.jl")
include("bayesian_inference.jl")
export BayesianInference, LinearInversion, PreAllocatedLinearInversion, NormalEquations,
    MaximumLikelihood, prediction, get_probs, get_probs!

using PrecompileTools: @setup_workload, @compile_workload
include("precompile.jl")

end