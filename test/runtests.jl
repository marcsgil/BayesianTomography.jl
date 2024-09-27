using BayesianTomography, Random, Test, LinearAlgebra
Random.seed!(123)

include("tomography_tests.jl")

@testset "Gell-Mann tests" begin
    include("gell_mann_tests.jl")
end