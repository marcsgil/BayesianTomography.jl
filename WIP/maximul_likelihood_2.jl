using LinearAlgebra, BayesianTomography, Optim

function probability(Π, ψ)
    real(dot(ψ, Π, ψ))
end

function likelihood(ψ, frequencies, povm)
    result = zero(eltype(frequencies))
    N = sum(abs2, ψ)
    for (n, f) ∈ enumerate(frequencies)
        result += (probability(povm[n], ψ) / N)^f
    end
    result
end
##
bs_povm = [[1.0 0im; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
quarter_wave_plate = [1 im; im 1] / √2

povm = augment_povm(bs_povm, half_wave_plate, quarter_wave_plate, weights=[1 / 2, 1 / 4, 1 / 4])
problem = StateTomographyProblem(povm)

ψ_true = sample(HaarVector(2))

outcomes = simulate_outcomes(ψ_true, povm, 10^2)
frequencies = normalize(outcomes, 1)

ψ0 = [1, 0.0im]

sol = optimize(ψ -> -likelihood(ψ, frequencies, povm), ψ0, LBFGS())

abs2(sol.minimizer ⋅ ψ_true) / sum(abs2, sol.minimizer)
##
using BayesianTomography, HDF5, ProgressMeter, FiniteDiff

includet("Data/data_treatment_utils.jl")
includet("Utils/position_operators.jl")
includet("Utils/basis.jl")

file = h5open("Data/Processed/pure_photocount.h5")

direct_lims = read(file["direct_lims"])
converted_lims = read(file["converted_lims"])
direct_x, direct_y = get_grid(direct_lims, (64, 64))
converted_x, converted_y = get_grid(converted_lims, (64, 64))
##
order = 2

lower = zeros(2 * order)
upper = vcat(fill(π / 2, order), fill(2π, order))

histories = file["histories_order$order"] |> read
coefficients = read(file["labels_order$order"])

basis = fixed_order_basis(order, [0, 0, 1 / √2, 1])

direct_operators = assemble_position_operators(direct_x, direct_y, basis)
mode_converter = diagm([cis(Float32(k * π / 2)) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
problem = StateTomographyProblem(operators)
##
m = 6
outcomes = complete_representation(History(view(histories, 1:2048, m)), (64, 64, 2))
#outcomes = simulate_outcomes(coefficients[:, m], operators, 2048)

J = findall(x -> x > 0, outcomes)
frequencies = normalize(outcomes[J], 1)
ψ = similar(coefficients[:, m])
p = (frequencies, operators[J], ψ)
angles_0 = fill(π / 4, 2 * order)
reduced_operators = operators[J]

ψ0 = coefficients[:, m]

sol = optimize(ψ -> -likelihood(ψ, frequencies, reduced_operators), ψ0, LBFGS())

pred = sol.minimizer
normalize!(pred)


likelihood(ψ0, frequencies, reduced_operators), likelihood(pred, frequencies, reduced_operators)

abs2(pred ⋅ ψ0)
##
method = BayesianInference(problem)

ρ_pred, _, _ = prediction(outcomes, method)
ψ_pred = project2pure(ρ_pred)

real(dot(coefficients[:, m], ρ_pred, coefficients[:, m]))
abs2(ψ_pred ⋅ coefficients[:, m])