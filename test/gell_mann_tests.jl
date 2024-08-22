import BayesianTomography: nth_off_diagonal
@test nth_off_diagonal(1) == (1, 2)
@test nth_off_diagonal(2) == (1, 3)
@test nth_off_diagonal(3) == (2, 3)
@test nth_off_diagonal(4) == (1, 4)
@test nth_off_diagonal(5) == (2, 4)
@test nth_off_diagonal(6) == (3, 4)
@test nth_off_diagonal(7) == (1, 5)
@test nth_off_diagonal(8) == (2, 5)
@test nth_off_diagonal(9) == (3, 5)
@test nth_off_diagonal(10) == (4, 5)

dims = 10:10:100

for dim ∈ dims
    θs = Vector{Float32}(undef, dim^2 - 1)
    ωs = GellMannMatrices(dim)
    σ = Matrix{ComplexF32}(undef, dim, dim)
    X = rand(ComplexF32, dim, dim)
    hermitianpart!(X)

    gell_mann_projection!(θs, X)
    @test [real(tr(X * ω)) for ω in ωs] ≈ θs
    @test sum(prod, zip(θs, ωs)) .+ tr(X) * I(dim) ./ dim ≈ X

    gell_mann_reconstruction!(σ, θs)
    @test σ ≈ X - tr(X) * I(dim) ./ dim

    ρ = X / tr(X)
    gell_mann_projection!(θs, ρ)
    density_matrix_reconstruction!(σ, θs)
    @test σ ≈ ρ
end