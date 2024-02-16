using HDF5, CairoMakie, LinearAlgebra, Tullio, Optim, BayesianTomography, ProgressMeter

function loss(pars, gaussian)
    xmin = pars[1]
    ymin = pars[2]
    xmax = pars[3]
    ymax = pars[4]
    x = LinRange(xmin, xmax, size(gaussian, 1))
    y = LinRange(ymin, ymax, size(gaussian, 2))
    @tullio proposal[i, j] := exp(-x[i]^2 - y[j]^2)
    mapreduce((x, y) -> (x - y)^2, +, normalize(gaussian .- pars[5]), normalize(proposal))
end

function get_grid_and_bg(calibration, pars0=[-4.0, -4.0, 4.0, 4.0, 0.0])
    direct_pars = optimize(pars -> loss(pars, calibration[:, :, 1]), pars0).minimizer
    converted_pars = optimize(pars -> loss(pars, calibration[:, :, 2]), pars0).minimizer

    direct_x = LinRange(direct_pars[1], direct_pars[3], size(calibration, 1))
    direct_y = LinRange(direct_pars[2], direct_pars[4], size(calibration, 2))
    converted_x = LinRange(converted_pars[1], converted_pars[3], size(calibration, 1))
    converted_y = LinRange(converted_pars[2], converted_pars[4], size(calibration, 2))

    backgrounds = [round(UInt8, direct_pars[5]), round(UInt8, converted_pars[5])]

    direct_x, direct_y, converted_x, converted_y, backgrounds
end

function build_basis(xd, yd, xc, yc, order, angle)
    dbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(xd, yd))
        for n ∈ 0:order])
    cbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)) * cis(-angle * n), Iterators.product(xc, yc))
        for n ∈ 0:order])
    stack([dbasis, cbasis], dims=3)
end

relu(x, y) = x > y ? x - y : zero(eltype(x))

function remove_background!(images, backgrounds)
    for (n, image) ∈ enumerate(eachslice(images, dims=3))
        map!(x -> relu(x, backgrounds[n]), image, image)
    end
end

function fidelity(ρ, σ)
    sqrt_ρ = sqrt(ρ)
    abs2(tr(sqrt(sqrt_ρ * σ * sqrt_ρ)))
end
##
order = 1
file = h5open("ExperimentalData/mixed_dataset.h5")
images = read(file["images_order$order"])
ρs = read(file["labels_order$order"])
calibration = read(file["calibration"])
close(file)

direct_x, direct_y, converted_x, converted_y, backgrounds = get_grid_and_bg(calibration)
remove_background!(images, backgrounds)

basis = build_basis(direct_x, direct_y, converted_x, converted_y, order, π / 6)
@tullio theo_images[x, y, m, n] := basis[x, y, m, j] * conj(basis[x, y, m, k]) * ρs[k, j, n] |> real
##
index = 15
astig = 1

fig = Figure(resolution=(1000, 500))
ax1 = CairoMakie.Axis(fig[1, 1],
    aspect=1)
ax2 = CairoMakie.Axis(fig[1, 2],
    aspect=1)
heatmap!(ax1, images[:, :, astig, index])
heatmap!(ax2, theo_images[:, :, astig, index])
fig
##
direct_operators = assemble_position_operators(direct_x, direct_y, order)
mode_converter = diagm([cis(k * π / 6) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
##
fids = Vector{Float64}(undef, size(images, 4))
mthd = LinearInversion(operators)
##
p = Progress(length(fids));
for n ∈ eachindex(fids)
    probs = normalize(images[:, :, :, n], 1)
    σ = prediction(probs, mthd)
    fids[n] = fidelity(ρs[:, :, n], σ)
    next!(p)
end
finish!(p)

fids
mean(fids)
##

probs = normalize(images[:, :, :, 1], 1)
σ = prediction(probs, mthd)
fidelity(ρs[:, :, 1], σ)