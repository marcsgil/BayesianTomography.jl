using BayesianTomography, LinearAlgebra, CairoMakie, ProgressMeter

order = 1
r = LinRange(-3, 3, 17)
position_operators = assemble_position_operators(r, r, order)
mode_converter = diagm([im^k for k ∈ 0:order])

transverse_Es = augment_povm(position_operators, mode_converter)
##
pbs_operators = [[1 0; 0 0], [0 0; 0 1]]
half_wave = [1 1; 1 -1] / √2
quarter_wave = [1 im; im 1] / √2

polarization_Es = augment_povm(pbs_operators, half_wave, quarter_wave)
##
ψ = sample_haar_vector(2)
outcomes = simulate_outcomes(ψ, transverse_Es, 1024)
pred_angles = prediction(outcomes, transverse_Es, MaximumLikelihood())

abs2(ψ ⋅ hurwitz_parametrization(pred_angles))
##
nobs = [round(Int, 2^n) for n in 5:12]
transverse_fids = Vector{Float64}(undef, length(nobs))
transverse_result = Vector{Float64}(undef, 100)
polarization_fids = similar(transverse_fids)
polarization_result = similar(transverse_result)
method = MaximumLikelihood()

@showprogress for k ∈ eachindex(nobs)
    Threads.@threads for n ∈ eachindex(transverse_result)
        ψ = sample_haar_vector(2)

        outcomes = simulate_outcomes(ψ, transverse_Es, nobs[k])
        pred_angles = prediction(outcomes, transverse_Es, method)
        transverse_result[n] = abs2(ψ ⋅ hurwitz_parametrization(pred_angles))

        outcomes = simulate_outcomes(ψ, polarization_Es, nobs[k])
        pred_angles = prediction(outcomes, polarization_Es, method)
        polarization_result[n] = abs2(ψ ⋅ hurwitz_parametrization(pred_angles))
    end

    transverse_fids[k] = mean(transverse_result)
    polarization_fids[k] = mean(polarization_result)
end
transverse_fids
polarization_fids
##
fig = Figure()
ax = Axis(fig[1, 1],
    xscale=log2,
    yscale=log10,
    xticks=[2^n for n in 1:13],
    xlabel="Number of observations",
    ylabel="Infidelity",
    title="Maximum Likelihood")
lines!(nobs, 1 .- transverse_fids, label="Transverse Structure (Order 1)")
lines!(nobs, 1 .- polarization_fids, label="Polarization")
axislegend(ax)
fig
#save("comparison.png", fig)