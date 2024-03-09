struct LinearInversion{T1,T2}
    pseudo_inv::Matrix{T1}
    basis::Vector{Matrix{T2}}
    function LinearInversion(povm)
        d = size(first(povm), 1)
        basis = get_hermitian_basis(d)
        A = [real(E ⋅ Ω) for E ∈ vec(povm), Ω ∈ vec(basis)]
        pseudo_inv = pinv(A)
        T = eltype(pseudo_inv)
        new{T,complex(T)}(pseudo_inv, basis)
    end
end

function prediction(outcomes, method::LinearInversion)
    vec_outcomes = vec(outcomes)
    xs = method.pseudo_inv * vec_outcomes
    ρ = sum(x * Ω for (x, Ω) ∈ zip(xs, method.basis)) |> project2density
end