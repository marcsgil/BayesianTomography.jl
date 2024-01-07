using BayesianTomography, LinearAlgebra, Logging
using LogDensityProblems, StaticArrays
Logging.disable_logging(Logging.Info)

order = 5
r = LinRange(-5, 5, 9)
position_operators = assemble_position_operators(r, r, order)
mode_converter = diagm([im^k for k ∈ 0:order])

operators = augment_povm(position_operators, mode_converter)
#operators = SMatrix{order + 1,order + 1}.(operators)
##
ψ = sample_haar_vector(order + 1)
outcomes = simulate_outcomes(ψ, operators, 1024)
##
ℓ = PureLogPosterior(outcomes, operators)
angles = ones(2 * (ℓ.dim - 1))
sangles = @SVector ones(2 * (ℓ.dim - 1))
θ = ones(ℓ.dim - 1)
ϕ = ones(ℓ.dim - 1)

@benchmark log_likellyhood($outcomes, $operators, $θ, $ϕ)
@benchmark log_likellyhood($outcomes, $operators, $θ, $ϕ)
##
using Symbolics, LinearAlgebra, StaticArrays, BayesianTomography

A = hermitianpart(rand(ComplexF64, 2, 2))

@variables θ::Real ϕ::Real

ψ = hurwitz_parametrization(θ, ϕ)

to_compute = real(dot(ψ, A, ψ))
f_expr = build_function(to_compute, [θ ϕ])
myf = eval(f_expr)

angles = SA[2.0, 3.0]
@benchmark myf($angles)
##
ψ = hurwitz_parametrization(2.0, 3.0)
@benchmark real(dot($ψ, $A, $ψ))