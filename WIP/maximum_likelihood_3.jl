using LinearAlgebra, BayesianTomography, Distributions

function simplified_pgd_qst(problem, outcomes, θ₀, step_size; max_iter=1000)
    vec_outcomes = vec(outcomes)
    traceless_part = problem.traceless_part
    trace_part = problem.trace_part
    buffer1 = similar(trace_part)
    buffer2 = similar(trace_part)
    ρ = density_matrix_reconstruction(θ₀)

    ℓπ_function!(∇ℓπ, θ) = BayesianTomography.log_likelihood!(∇ℓπ, buffer1, buffer2, vec_outcomes, traceless_part, trace_part, θ)

    θ = zeros(eltype(problem.traceless_part), size(problem.traceless_part, 2))

    for _ in 1:max_iter
        ∇ℓπ(θ)
        θ .+= step_size * ∇ℓπ(θ)
        density_matrix_reconstruction!(ρ, θ)
        project2density!(ρ)
        gell_mann_projection!(θ, ρ)
    end
end
##

H = [1, 0.0im]
V = [0.0im, 1]
D = [1 + 0im, 1] / √2
A = [1 + 0im, -1] / √2
R = [1, im] / √2
L = [1, -im] / √2

h = H * H'
v = V * V'
d = D * D'
a = A * A'
r = R * R'
l = L * L'

povm = [h, v, d, a, r, l] / 3
sum(povm)
##
θ = [1 / √2, 0, 0]
θ₀ = zero(θ)
ρ = density_matrix_reconstruction(θ)

probs = [real(tr(Π * ρ)) for Π in povm]
Categorical()


#outcomes = simulate_outcomes(ρ, povm, 10^5)
#frequencies = normalize(outcomes, 1)

@code_warntype simplified_pgd_qst(∇F, ρ₀, 0.01, povm, probs; max_iter=1000)

ρ_pred = simplified_pgd_qst(∇F, ρ₀, 0.01, povm, probs; max_iter=10000)


ρ
ρ_pred

F(ρ_pred, povm, probs) / F(ρ, povm, probs)

[real(tr(Π * ρ_pred)) for Π in povm]