using CSV, LinearAlgebra

file = CSV.File("WIP/CountsForAndre.csv", header=false)

parse_c(s) = parse(ComplexF32, s)

outcomes = [Float32(parse_c(row[4])) for row in file]
frequencies = normalize(outcomes, 1)
ψ1 = [[parse_c(row[5]), parse_c(row[6])] for row ∈ file]
ψ2 = [[parse_c(row[7]), parse_c(row[8])] for row ∈ file]

povm = [kron(pair[1] * pair[1]', pair[2] * pair[2]') for pair in zip(ψ1, ψ2)]

using BayesianTomography

problem = StateTomographyProblem(povm)

method = MaximumLikelihood(problem)

ρ_pred, θ_pred = prediction(outcomes, method);

ρ_pred

ψ_true = [1 + 0im, 0, 0, 1] / √2
ρ_true = ψ_true * ψ_true'

fidelity(ρ_true, ρ_pred)
##
@benchmark prediction($outcomes, $method)