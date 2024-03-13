using BayesianTomography, Random, Test
Random.seed!(1234)

@testset "Representation tests" begin
    include("representation_tests.jl")
end

@testset "Tomography tests" begin
    include("tomography_tests.jl")
end