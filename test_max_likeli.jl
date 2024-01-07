using BayesianTomography, LinearAlgebra, ProgressMeter

order = 1
r = LinRange(-4.5, 4.5, 17)
position_operators = assemble_position_operators(r, r, order)
mode_converter = diagm([im^k for k ∈ 0:order])

operators = augment_povm(position_operators, mode_converter)
ψ = sample_haar_vector(order + 1)
outcomes = simulate_outcomes(ψ, operators, 2^15)

predicted_ang = prediction(outcomes, operators, MaximumLikelihood())
ψ_pred = hurwitz_parametrization(predicted_ang)
abs2(ψ ⋅ ψ_pred)
##
orders = 1:4
fids = Matrix{Float64}(undef, 100, length(orders))

for n ∈ axes(fids, 2)
    order = orders[n]
    R = 2.5 + 0.5 * order
    r = LinRange(-R, R, 17)
    position_operators = assemble_position_operators(r, r, order)
    mode_converter = diagm([im^k for k ∈ 0:order])
    operators = augment_povm(position_operators, mode_converter)
    p = Progress(size(fids, 1))
    Threads.@threads for m ∈ axes(fids, 1)
        ψ = sample_haar_vector(order + 1)
        outcomes = simulate_outcomes(ψ, operators, 2^15)
        predicted_ang = prediction(outcomes, operators, MaximumLikelihood(); ntries=8)
        ψ_pred = hurwitz_parametrization(predicted_ang)
        fids[m, n] = abs2(ψ ⋅ ψ_pred)
        next!(p)
    end
    finish!(p)
end

mean(fids, dims=1)
##
using HDF5, CairoMakie, Tullio

order = 1
file = h5open("Datasets/pure_dataset.h5")
cs = read(file["coefficients_order1"])
cs = cs[1:2, :] + im * cs[3:4, :]
images = read(file["images_order1"])
close(file)

r = LinRange(-2.5, 2.5, 512)
basis = stack([
    map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(r, r))
    for n ∈ 0:order])
@tullio timage[x, y, image] := cs[j, image] * basis[x, y, j] |> abs2
##
index = 1

fig = Figure(resolution=(1000, 500))
ax1 = Axis(fig[1, 1],
    aspect=1)
ax2 = Axis(fig[1, 2],
    aspect=1)
heatmap!(ax1, images[:, :, 1, index])
heatmap!(ax2, timage[:, :, index])
fig
