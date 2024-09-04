using LinearAlgebra

function project2density(μ)
    F = eigen(μ, sortby=λ -> -real(λ))
    vals = real.(F.values)
    vecs = F.vectors
    λs = similar(vals)
    d = length(vals)

    accumulator = zero(real(eltype(μ)))
    for i ∈ d:(-1):1
        if vals[i] + accumulator / i ≥ 0
            for j ∈ 1:i
                λs[j] = vals[j] + accumulator / i
            end
            break
        else
            λs[i] = 0
            accumulator += vals[i]
        end
    end

    vecs * Diagonal(λs) * vecs'
end

μ = rand(ComplexF64, 2, 2)
μ /= tr(μ)
@code_warntype project2density(μ)
eigvals(project2density(μ))

function ∇ℓ(ρ, counts, povm)
    ∇ℓ = zero(ρ)
    for (count, Π) in zip(counts, povm)
        prob = real(tr(Π * ρ))
        change = Π * (count / prob)
        if all(isfinite, change)
            ∇ℓ += change
        end
    end
    ∇ℓ - tr(∇ℓ) * I / size(ρ, 1)
end

function gradient_ascent(counts, povm, ρ₀, step_size, max_iter)
    ρ = copy(ρ₀)

    for i in 1:max_iter
        grad = ∇ℓ(ρ, counts, povm)
        ρ += step_size * grad
        ρ = project2density(ρ)
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

povm = [h, v, d, a, r, l] / 3
sum(povm)
##
povm2 = [kron(Π1, Π2) for Π1 in povm, Π2 in povm]
##
dim = 4
X = rand(ComplexF64, dim, dim)
ρ = X' * X
ρ ./= tr(ρ)
ρ₀ = Matrix(I(dim) / (dim + 0im))

probs = [real(tr(Π * ρ)) for Π in povm2]
counts = [round(Int, prob * 10^3) for prob in probs]
frequencies = normalize(counts, 1)

#@code_warntype gradient_ascent(frequencies, povm2, ρ₀, 0.01, 1000)

ρ_pred = gradient_ascent(frequencies, povm2, ρ₀, 0.01, 500)
real(tr(sqrt(ρ * ρ_pred)))
##
@benchmark gradient_ascent($frequencies, $povm2, $ρ₀, 0.01, 500)