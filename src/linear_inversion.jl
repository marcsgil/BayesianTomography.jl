function A_matrix(j, d)
    @assert j ≤ d - 1
    diag = zeros(d)
    for k ∈ 1:j
        diag[k] = 1
    end
    diag[j+1] = -j
    result = diagm(diag)
    normalize!(result)
    hermitianpart(result)
end

function B_matrix(j, k, d)
    result = zeros(d, d)
    result[j, k] = 1
    result[k, j] = 1
    normalize!(result)
    hermitianpart(result)
end

function C_matrix(j, k, d)
    result = zeros(ComplexF64, d, d)
    result[j, k] = im
    result[k, j] = -im
    normalize!(result)
    hermitianpart(result)
end

function triangular_indices(d)
    indices = Vector{Tuple{Int,Int}}(undef, d * (d - 1) ÷ 2)
    counter = 0
    for j ∈ 1:d
        for k ∈ 1:j-1
            counter += 1
            indices[counter] = (j, k)
        end
    end
    indices
end

function get_basis(d)
    I_d = Matrix{Float64}(I, d, d)
    Js = triangular_indices(d)
    As = [A_matrix(j, d) for j ∈ 1:d-1]
    Bs = [B_matrix(J..., d) for J ∈ Js]
    Cs = [C_matrix(J..., d) for J ∈ Js]
    vcat([I_d / √d], As, Bs, Cs)
end

struct LinearInversion{T1,T2}
    pseudo_inv::Matrix{T1}
    basis::Vector{Matrix{T2}}
    function LinearInversion(povm)
        d = size(first(povm), 1)
        basis = get_basis(d)
        A = [real(tr(E * Ω)) for E ∈ vec(povm), Ω ∈ vec(basis)]
        pseudo_inv = pinv(A)
        T = eltype(pseudo_inv)
        new{T,complex(T)}(pseudo_inv, basis)
    end
end

function project(ρ)
    F = eigen(hermitianpart(ρ))
    λs = [λ > 0 ? λ : 0 for λ ∈ real.(F.values)]
    normalize!(λs, 1)
    sum(λ * v * v' for (λ, v) ∈ zip(λs, eachcol(F.vectors)))
end

function prediction(outcomes, method::LinearInversion)
    vec_outcomes = vec(outcomes)
    xs = method.pseudo_inv * vec_outcomes
    ρ = sum(x * Ω for (x, Ω) ∈ zip(xs, method.basis)) |> project2density
end