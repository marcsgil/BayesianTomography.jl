using BayesianTomography, LinearAlgebra

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

measurements = [h, v, d, r]

problem = StateTomographyProblem(measurements)
get_probabilities(problem, [0, 0, 1 / √2])

cond(problem)