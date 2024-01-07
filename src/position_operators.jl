function hg(r, p)
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
    N * [(hermiteh(m1, r[1]) * hermiteh(order - m1, r[2]) * hermiteh(m2, r[1]) * hermiteh(order - m2, r[2])
          /
          √prod(factorial, (m1, order - m1, m2, order - m2))) for m1 ∈ 0:order, m2 ∈ 0:order]
end

function assemble_position_operators(x, y, order)
    operators = Matrix{Matrix{Float64}}(undef, length(x) - 1, length(y) - 1)

    for i ∈ 1:length(x)-1, j ∈ 1:length(y)-1
        prob = IntegralProblem(hermite_position_operator, [x[i], y[j]], [x[i+1], y[j+1]], order)
        operators[i, j] = solve(prob, HCubatureJL()).u
    end

    operators
end