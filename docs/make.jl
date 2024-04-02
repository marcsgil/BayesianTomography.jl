using Documenter, BayesianTomography, LinearAlgebra

ENV["GKSwstype"] = "100"
using Plots
Plots.scalefontsizes(1.5)

makedocs(
    sitename="BayesianTomography.jl",
    pages=[
        "index.md",
        "usage.md",
        "theory.md",
        "api.md"])

deploydocs(
    repo="https://github.com/marcsgil/BayesianTomography.jl",
)