using CairoMakie, BayesianTomography, LinearAlgebra

Es = hermitianpart.([outer_product([1, 0]) / 3, outer_product([0, 1]) / 3,
    outer_product([1, 1]) / 6, outer_product([1, -1]) / 6,
    outer_product([1, im]) / 6, outer_product([1, -im]) / 6])
##
nobs = [round(Int, 2^n) for n in 1:13]
fids = Vector{Float64}(undef, length(nobs))
result = Vector{Float64}(undef, 1000)

for k ∈ eachindex(nobs)
    println("$k")
    for n ∈ eachindex(result)
        ψ = sample_haar_vector(2)
        m = simulate_outcomes(ψ, Es, nobs[k])

        θ, ϕ = prediction(m, 3000, nchains=4)
        result[n] = abs2(ψ ⋅ hurwitz_parametrization(θ, ϕ))
    end

    fids[k] = mean(result)
end

fids
##
fig = Figure()
ax = Axis(fig[1, 1],
    xscale=log2,
    yscale=log10,
    xticks=[2^n for n in 1:13],
    xlabel="Number of observations",
    ylabel="Infidelity",)
lines!(nobs, 1 .- fids)
fig
##