using LinearAlgebra, BayesianTomography

function ℓ(ρ, counts, povm)
    ℓ = zero(real(eltype(ρ)))
    for (count, Π) in zip(counts, povm)
        prob = real(Π ⋅ ρ)
        ℓ += count * log(prob)
    end
    ℓ
end

function ∇ℓ(ρ, counts, povm)
    ∇ℓ = zero(ρ)
    for (count, Π) in zip(counts, povm)
        prob = real(Π ⋅ ρ)
        change = Π * (count / prob)
        if all(isfinite, change)
            ∇ℓ += change
        end
    end
    ∇ℓ - tr(∇ℓ) * I / size(ρ, 1)
end

function gradient_ascent(counts, povm, ρ₀, t, γ, max_iter, tol=1e-12)
    ρ = copy(ρ₀)
    candidate = similar(ρ)

    for i in 1:max_iter
        grad = ∇ℓ(ρ, counts, povm)
        @. candidate = ρ + t * grad
        project2density!(candidate)
        δ = ρ - candidate

        likelihood = ℓ(ρ, counts, povm)
        likelihood_candidate = ℓ(candidate, counts, povm)

        ti = t

        while likelihood ≤ likelihood_candidate + real(grad ⋅ δ - δ ⋅ δ / (2ti))
            ti *= γ
            @. candidate = ρ + t * grad
            project2density!(candidate)
            δ .= ρ - candidate
            likelihood_candidate = ℓ(candidate, counts, povm)
        end

        sum(abs2, δ) < tol && return ρ

        copy!(ρ, candidate)
    end

    ρ
end
##
H = [1, 0.0im]
V = [0.0im, 1]
D = [1 + 0im, 1] / √2
A = [1 + 0im, -1] / √2
R = [1, -im] / √2
L = [1, im] / √2

h = H * H'
v = V * V'
d = D * D'
a = A * A'
r = R * R'
l = L * L'

povm = [kron(h, h),
    kron(h, v),
    kron(v, v),
    kron(v, h),
    kron(r, h),
    kron(r, v),
    kron(d, v),
    kron(d, h),
    kron(d, r),
    kron(d, d),
    kron(r, d),
    kron(h, d),
    kron(v, d),
    kron(v, l),
    kron(h, l),
    kron(r, l),]

ψ_true = [1 + 0im, 0, 0, 1] / √2
ρ_true = ψ_true * ψ_true'
θ_true = gell_mann_projection(ρ_true)

outcomes = [34749, 324, 35805, 444, 16324, 17521, 13441,
    16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586]

problem = StateTomographyProblem(povm)
frequencies = normalize(outcomes, 1)

dim = 4
ρ₀ = Matrix(I(dim) / (dim + 0im))
ρ_pred = gradient_ascent(frequencies, problem.effective_povm, ρ₀, 0.1, 0.5, 10^5)
ρ_pred
##

problem.inv_kraus_operator

BayesianTomography.post_measurement_state!(ρ_pred, problem.inv_kraus_operator)


ρ_pred
fidelity(ρ_true, ρ_pred)
##
povm2 = [kron(Π1, Π2) for Π1 in povm, Π2 in povm]

problem = StateTomographyProblem(povm2)
##
dim = 4
X = rand(ComplexF64, dim, dim)
ρ = X' * X
ρ ./= tr(ρ)
ρ₀ = Matrix(I(dim) / (dim + 0im))
##
probs = [real(tr(Π * ρ)) for Π in problem.effective_povm]
outcomes = simulate_outcomes(probs, 10^6)
##


#@code_warntype gradient_ascent(frequencies, povm2, ρ₀, 0.01, 1000)
##

##
@benchmark gradient_ascent($frequencies, $problem.effective_povm, ρ₀, 0.1, 0.5, 10^5)

ρ1 = sample(GinibreEnsamble(2))
ρ2 = sample(GinibreEnsamble(2))

tr(ρ1 * ρ2)

ρ1 ⋅ ρ2