using Documenter, BayesianTomography

ENV["GKSwstype"] = "100"
using Plots
pgfplotsx()
Plots.scalefontsizes(1.5)

makedocs(
    sitename="BayesianTomography.jl",
    pages=[
        "index.md",
        "usage.md",
        "theory.md",
        "api.md"])