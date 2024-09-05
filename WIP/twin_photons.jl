using LinearAlgebra, BayesianTomography

H = [1, 0.0im]
V = [0.0im, 1]
D = (H + V) / √2
A = (H - V) / √2
R = (H - im * V) / √2
L = (H + im * V) / √2

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

method = MaximumLikelihood(problem)
ρ_pred, θ_pred = prediction(outcomes, method);

#method = BayesianInference(problem)
#ρ_pred, θs, cov = prediction(outcomes, method; nsamples=10^6, nwarm=10^5);

eigvals(ρ_pred)
##
fidelity(ρ_true, ρ_pred)
##
ρ_article = [
    0.5069 -0.0239+0.0106im -0.0412-0.0221im 0.4833+0.0329im;
    -0.0239-0.0106im 0.0048 0.0023+0.0019im -0.0296-0.0077im;
    -0.0412+0.0221im 0.0023-0.0019im 0.0045 -0.0425+0.0192im;
    0.4833-0.0329im -0.0296+0.0077im -0.0425-0.0192im 0.4839
]

fidelity(ρ_article, ρ_true)
##
θ_article = gell_mann_projection(ρ_article)
buffer = similar(frequencies)

BayesianTomography.log_likelihood!(buffer, frequencies, problem.traceless_part, problem.trace_part, θ_article)
BayesianTomography.log_likelihood!(buffer, frequencies, problem.traceless_part, problem.trace_part, θ_pred)
BayesianTomography.log_likelihood!(buffer, frequencies, problem.traceless_part, problem.trace_part, θ_true)
##
@time for n ∈ 1:10
    prediction(frequencies, method)
end

@benchmark prediction($frequencies, $method)