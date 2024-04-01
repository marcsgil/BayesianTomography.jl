using FiniteDifferences, LinearAlgebra

# Fidelity between two density matrices
function fidelity(ρ::AbstractMatrix, σ::AbstractMatrix)
    sqrt_ρ = sqrt(ρ)
    abs2(tr(sqrt(sqrt_ρ * σ * sqrt_ρ)))
end

# Matrix representation of a Bloch vector
function matrix_representation(r)
    [(1+r[1]) (r[2]-r[3]im); (r[2]+r[3]im) (1-r[1])] ./ 2
end

# Fidelity between a density matrix and a Bloch vector
function fidelity(ρ::AbstractMatrix, r::AbstractVector)
    fidelity(ρ, matrix_representation(r))
end

# Gradient of the fidelity
function ∇fidelity(ρ::AbstractMatrix, r::AbstractVector)
    grad(central_fdm(5, 1), r -> fidelity(ρ, r), r)[1]
end

r = [1 / 4, 1 / 3, 1 / 2]
ρ = matrix_representation([1 / 3, 1 / 4, 1 / 2])
∇fidelity(ρ, r)
##
function matrix_evalpoly(x, p)
    d = size(x, 1)
    y = zero(x)

    for k ∈ 1:d
        y[k, k] = p[end]
    end

    cache = copy(y)

    for i in length(p)-1:-1:1
        mul!(cache, y, x)
        copy!(y, cache)
        for k ∈ 1:d
            y[k, k] += p[i]
        end
    end
    return result
end

function LinearAlgebra.sqrt(A::AbstractMatrix, N)
    """nA = norm(A)
    T = eltype(A)
    p = map(k -> k == 0 ? zero(T) : abs(T(binomial(1 / 2, k))), 0:N)
    id = Matrix{eltype(A)}(I, size(A)...)
    √nA .* (id .- matrix_evalpoly(id .- A ./ nA, p))"""

    B = Matrix{eltype(A)}(I, size(A)...)
    for _ in 1:N
        B .= (B .+ A / B) ./ 2
    end

    B
end

A = rand(ComplexF32, 5, 5)
B = A * A'

@b sqrt($B, 20)
@b sqrt($B)

sqrt(B, 100)
sqrt(B)

p = [1, 2, 3]
@b matrix_evalpoly($ρ, $p)



binomial(1 / 2, 0)
I + 2ρ + 3ρ^2