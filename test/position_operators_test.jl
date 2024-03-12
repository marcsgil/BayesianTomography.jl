using ClassicalOrthogonalPolynomials, Tullio, LinearAlgebra, Random, OnlineStats
Random.seed!(1234)  # Set the seed for reproducibility

function hg(x, y, m, n)
    N = 1 / sqrt(2^(m + n) * factorial(m) * factorial(n) * π)
    N * hermiteh(m, x) * hermiteh(n, y) * exp(-(x^2 + y^2) / 2)
end

function transverse_basis(order)
    [(r, par) -> hg(r[1], r[2], order - n, n) for n ∈ 0:order]
end

function transverse_basis(xd, yd, xc, yc, order, angle)
    basis = Array{complex(eltype(xd))}(undef, length(xd), length(yd), 2, order + 1)

    @tullio basis[i, j, 1, k] = hg(xd[i], yd[j], order - k + 1, k - 1)
    @tullio basis[i, j, 2, k] = hg(xc[i], yc[j], order - k + 1, k - 1)

    for k ∈ 0:order
        basis[:, :, 2, k+1] .*= cis(k * angle)
    end

    basis
end

function label2image!(dest, c::AbstractVector, basis)
    @tullio dest[i, j, m] = basis[i, j, m, k] * c[k] |> abs2
end

function label2image(c::AbstractVector, r, angle)
    basis = transverse_basis(r, r, r, r, size(c, 1) - 1, angle)
    image = Array{Float32,3}(undef, length(r), length(r), 2)
    label2image!(image, c, basis)
    image
end

function label2image!(dest, ρ::AbstractMatrix, basis)
    @tullio dest[i, j, k] = ρ[m, n] * basis[i, j, k, m] * conj(basis[i, j, k, n]) |> real
end

function label2image(ρ::AbstractMatrix, r, angle)
    basis = transverse_basis(r, r, r, r, size(ρ, 1) - 1, angle)
    image = Array{Float32,3}(undef, length(r), length(r), 2)
    label2image!(image, ρ, basis)
    image
end

@testset "Linear Inversion (Position Operators)" begin
    for order ∈ 1:4
        @info "Testing Order $order"

        basis = transverse_basis(order)
        R = 2.5 + 0.5 * order
        rs = LinRange(-R, R, 64)
        direct_operators = assemble_position_operators(rs, rs, basis)
        mode_converter = diagm([cis(k * π / (order + 1)) for k ∈ 0:order])
        astig_operators = assemble_position_operators(rs, rs, basis)
        unitary_transform!(astig_operators, mode_converter)
        operators = compose_povm(direct_operators, astig_operators)
        mthd1 = LinearInversion(operators)
        mthd2 = BayesianInference(operators, 10^5, 10^3)
        hermitian_basis = get_hermitian_basis(order + 1)

        N = 5
        for n ∈ 1:N
            X = randn(ComplexF64, order + 1, order + 1)
            ρ = X * X' / tr(X * X')
            images = label2image(ρ, rs, π / (order + 1))
            @test fidelity(prediction(images, mthd1), ρ) ≥ 0.995

            normalize!(images, 1)
            simulate_outcomes!(images, 2^15)
            outcomes = array2dict(images)
            σ = linear_combination(mean(prediction(outcomes, mthd2)), hermitian_basis)
            @test fidelity(σ, ρ) ≥ 0.99
        end
    end
end