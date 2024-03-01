"""function hg(r, p)
    m, n = p
    N = √(π * 2^(m + n) * factorial(m) * factorial(n))
    hermiteh(m, r[1]) * hermiteh(n, r[2]) * exp(sum(-r .^ 2) / 2) / N
end

function hg_product(r, p)
    m1, n1, m2, n2 = p
    hg(r, (m1, n1)) * hg(r, (m2, n2))
end

function hermite_position_operator(r, order)
    N = exp(-sum(abs2, r)) / (π * 2^order)
    N * [(hermiteh(order - m1, r[1]) * hermiteh(m1, r[2]) * hermiteh(order - m2, r[1]) * hermiteh(m2, r[2])
          /
          √prod(factorial, (m1, order - m1, m2, order - m2))) for m1 ∈ 0:order, m2 ∈ 0:order]
end"""

function assemble_position_operators(xs, ys, basis)
    operators = Matrix{Matrix{ComplexF64}}(undef, length(xs), length(ys))

    Δx = (xs[2] - xs[1]) / 2
    Δy = (ys[2] - ys[1]) / 2

    function integrand!(y, r, par)
        for k ∈ eachindex(basis), j ∈ eachindex(basis)
            y[j, k] = conj(basis[j](r, par)) * basis[k](r, par)
        end
    end

    prototype = zeros(ComplexF64, length(basis), length(basis))
    f = IntegralFunction(integrand!, prototype)

    Threads.@threads for n ∈ eachindex(ys)
        for m ∈ eachindex(xs)
            domain = [xs[m] - Δx, ys[n] - Δy], [xs[m] + Δx, ys[n] + Δy]
            prob = IntegralProblem(f, domain)
            operators[m, n] = solve(prob, HCubatureJL()).u
        end
    end

    operators
end