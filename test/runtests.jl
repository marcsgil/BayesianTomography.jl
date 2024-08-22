using BayesianTomography, Random, Test, LinearAlgebra
Random.seed!(1234)

"""@testset "Representation tests" begin
    include("representation_tests.jl")
end"""

@testset "Tomography tests" begin
    include("tomography_tests.jl")
end

"""@testset "Gell-Mann tests" begin
    include("gell_mann_tests.jl")
end"""