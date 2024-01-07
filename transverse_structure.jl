using CairoMakie, BayesianTomography, LinearAlgebra, ProgressMeter

order = 3
r = LinRange(-4.0, 4.0, 65)
position_operators = assemble_position_operators(r, r, order)
mode_converter = diagm([im^k for k ∈ 0:order])

operators = augment_povm(position_operators, mode_converter)
##
method = MetropolisHastings(; nchains=8)
ψ = sample_haar_vector(order + 1)
outcomes = simulate_outcomes(ψ, operators, 1024)
pred_angles = prediction(outcomes, operators, method)

abs2(ψ ⋅ hurwitz_parametrization(pred_angles))
##
nobs = [round(Int, 2^n) for n in 6:11]
fids = Vector{Float64}(undef, length(nobs))
min_fids = zeros(length(nobs))
result = Vector{Float64}(undef, 100)

for k ∈ eachindex(nobs)
    @show k
    progress = Progress(length(result))
    Threads.@threads for n ∈ eachindex(result)
        ψ = sample_haar_vector(order + 1)
        outcomes = simulate_outcomes(ψ, operators, nobs[k])

        angles = prediction(outcomes, operators, method)
        ψ_pred = hurwitz_parametrization(angles)
        result[n] = abs2(ψ ⋅ ψ_pred)
        next!(progress)
    end
    finish!(progress)
    fids[k] = mean(result)
    min_fids[k] = minimum(result)
end
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
fids
##
using HDF5
file = h5open("theoretical_fids.h5", "cw")
file["order$order"] = fids
close(file)
##
using HDF5
file = h5open("theoretical_fids.h5", "r")
nobs = file["nobs"][:]
fids = stack(file["order$order"][:] for order in 1:4)
close(file)

fig = Figure()
ax = Axis(fig[1, 1],
    xscale=log2,
    yscale=log10,
    xticks=[2^n for n in 1:13],
    xlabel="Number of observations",
    ylabel="Infidelity",)
series!(nobs, 1 .- fids', labels=["Order $order" for order in 1:4])
axislegend(ax)
fig
save("BayesianTomography.jl/plots/theoretical_fids.png", fig)