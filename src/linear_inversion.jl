struct LinearInversion{T1,T2}
    pseudo_inv::Matrix{T1}
    basis::Array{T2,3}
    function LinearInversion(povm)
        d = size(first(povm), 1)
        basis = get_hermitian_basis(d)
        pseudo_inv = stack(F -> real_representation(F, basis), povm, dims=1) |> pinv
        T = eltype(pseudo_inv)
        new{T,complex(T)}(pseudo_inv, basis)
    end
end

function prediction(outcomes, method::LinearInversion)
    xs = method.pseudo_inv * vec(outcomes)
    linear_combination(xs, method.basis) |> project2density
end