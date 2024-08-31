using LinearAlgebra, BayesianTomography, Optimization, OptimizationOptimJL, Optim

"""function pure_state_vector!(dest, θ, ϕ)
    s, c = sincos(θ[1])
    dest[1] = c
    sin_prod = s
    for n ∈ 2:length(θ)
        s, c = sincos(θ[n])
        dest[n] = sin_prod * c * cis(ϕ[n-1])
        sin_prod *= s
    end

    dest[end] = sin_prod * cis(ϕ[end])

    nothing
end"""

function pure_state_vector(theta, phi)
    # Check if the input vectors have the same length
    if length(theta) != length(phi)
        throw(ArgumentError("theta and phi vectors must have the same length"))
    end
    
    # Determine the dimension of the Hilbert space
    d = length(theta) + 1
    
    # Initialize the state vector
    psi = zeros(ComplexF64, d)
    
    # Set the first component
    psi[1] = cos(theta[1])
    
    # Calculate the remaining components
    for i in 2:d-1
        psi[i] = prod(sin, (@view theta[1:i-1])) * cos(theta[i]) * cis(phi[i-1])
    end
    
    # Set the last component
    if d > 1
        psi[d] = prod(sin, theta) * exp(im * phi[end])
    end
    
    # Normalize the state vector (to account for any numerical errors)
    return normalize(psi)
end

function pure_state_vector(θ, ϕ)
    T = complex(float(eltype(θ)))
    d = length(θ) + 1
    dest = Vector{T}(undef, d)
    pure_state_vector!(dest, θ, ϕ)
    dest
end

function probability(Π, ψ)
    real(dot(ψ, Π, ψ))
end

function likelihood(angles, frequencies, povm, ψ)
    θ = @view angles[1:end÷2]
    ϕ = @view angles[end÷2+1:end]
    result = zero(float(eltype(angles)))
    for (n, f) ∈ enumerate(frequencies)
        #pure_state_vector!(ψ, θ, ϕ)
        ψ = pure_state_vector(θ, ϕ)
        result += probability(povm[n], ψ)^f
    end
    result
end

function loss(angles, p)
    frequencies, povm, ψ = p
    -likelihood(angles, frequencies, povm, ψ)
end
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

lower = zeros(2*order)
upper = vcat(fill(π/2, order), fill(2π, order))

histories = file["histories_order$order"] |> read
coefficients = read(file["labels_order$order"])

basis = fixed_order_basis(order, [0, 0, 1 / √2, 1])

direct_operators = assemble_position_operators(direct_x, direct_y, basis)
mode_converter = diagm([cis(Float32(k * π / 2)) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
##
m = 2
outcomes = complete_representation(History(view(histories, 1:2048, m)), (64, 64, 2))

J = findall(x->x>0, outcomes)
frequencies = normalize(outcomes[J], 1)
ψ = similar(coefficients[:, m])
p = (frequencies, operators[J], ψ)
angles_0 = fill(π / 4, 2*order)
reduced_operators = operators[J]

sol = optimize(angles->-likelihood(angles, frequencies, reduced_operators, ψ), lower, upper, angles_0, SAMIN(), Optim.Options(iterations=10^6))
abs2(pure_state_vector(sol.minimizer[1:end ÷2], sol.minimizer[end ÷2 + 1:end]) ⋅ coefficients[:, m])

prob = StateTomographyProblem(operators)
mthd = BayesianInference(prob)

ρ_pred, θ_pred, cov = prediction(outcomes, mthd)

ψ = project2pure(ρ_pred)
abs2(ψ ⋅ coefficients[:, m])