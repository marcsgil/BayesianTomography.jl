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