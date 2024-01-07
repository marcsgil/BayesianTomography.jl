using BayesianTomography, LinearAlgebra, CairoMakie, ProgressMeter, Distributions

function formatter(value::Number)
    if value == 0
        return "0"
    elseif value ≈ π
        return L"\pi"
    elseif isinteger(value / π)
        return L"%$(Int(value/π)) \pi"
    else
        return L"%$(value/π) \pi"
    end
end

formatter(values::AbstractVector) = formatter.(values)
##
order = 1
r = LinRange(-3, 3, 9)
position_operators = assemble_position_operators(r, r, order)
mode_converter = diagm([im^k for k ∈ 0:order])

operators = hermitianpart.(augment_povm(position_operators, mode_converter))
##
ψ = sample_haar_vector(order + 1)
outcomes = simulate_outcomes(ψ, operators, 1024)

θ_pred, ϕ_pred = prediction(outcomes, operators, 2000, AdvancedHMCSampler())
ψ_pred = hurwitz_parametrization(θ_pred, ϕ_pred)
abs2(ψ ⋅ ψ_pred)
##
θs = LinRange(0, π, 256)
ϕs = LinRange(0, 2π, 256)

true_angles = (π / 2, π)
ψ = hurwitz_parametrization(true_angles...)
outcomes = simulate_outcomes(ψ, operators, 128)

log_posteriors = [log_likellyhood(outcomes, operators, θ, ϕ) + log_prior(θ, ϕ)
                  for θ in θs, ϕ in ϕs]
M = minimum(abs, filter(isfinite, log_posteriors))
posteriors = exp.(log_posteriors .+ M)
normalize!(posteriors, Inf)

samples = sample_posterior(outcomes, operators, 2000, AdvancedHMCSampler(); nadapts=1000, nchains=1)
n = size(first(operators), 1) - 1
sampled_θ = acos.(cos.(vec(samples.value[:, 1:n, :])))
sampled_ϕ = mod2pi.(vec(samples.value[:, n+1:2n, :]))

pred = prediction(outcomes, operators, MaximumLikelihood())

ψ_pred = hurwitz_parametrization(mean(sampled_θ), circular_mean(sampled_ϕ))
fid = abs2(ψ ⋅ ψ_pred)

fig = Figure()
ax = Axis(fig[1, 1],
    xticks=LinRange(0, π, 5),
    yticks=LinRange(0, 2π, 5),
    xtickformat=formatter,
    ytickformat=formatter,
    xlabel=L"\theta",
    ylabel=L"\phi",
    title="Fidelity = $fid")


hm = heatmap!(ax, θs, ϕs, log_posteriors, colormap=:jet)
scatter!(ax, sampled_θ, sampled_ϕ, color=:black, markersize=3)
Colorbar(fig[1, 2], hm, label="Rescaled Log Posterior")
scatter!(ax, true_angles[1], true_angles[2], color=:yellow, markersize=30, marker=:cross)
scatter!(ax, pred[1], pred[2], color=:red, markersize=30, marker=:cross)
#save("diagnosis/posterior4.png", fig, px_per_unit=2)
fig
##
mean(sampled_θ)
circular_mean(sampled_ϕ)
true_angles
##