using LinearAlgebra, BayesianTomography

outcomes = [34749, 324, 35805, 444, 16324, 17521, 13441,
    16901, 17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586]

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

g = sum(povm)

L = cholesky(g).L
invL = inv(L)

povm = [invL * Π * invL' for Π in povm]

prob = StateTomographyProblem(povm)
method = BayesianInference(prob)

ρ, θs, cov = prediction(outcomes, method; nsamples=10^6,
    nwarm=10^5,)

σ = invL' * ρ * invL
σ = σ / tr(σ)
#abs2.(project2pure(σ))

ψ_true = [1, 0, 0, 1] / √2
ρ_true = ψ_true * ψ_true'
fidelity(ρ_true, σ)
##
round.(σ, digits=3)

tr(σ^2)
