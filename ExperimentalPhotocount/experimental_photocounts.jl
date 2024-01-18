using TiffImages, CairoMakie, Statistics, BayesianTomography, Optim, Tullio, LinearAlgebra
using HDF5, ProgressMeter

get_value(x) = x.val

function extract_counts(frames, background, threshold; make_movie=false)
    μ = dropdims(mean(background, dims=3), dims=3)
    σ = dropdims(std(background, dims=3), dims=3)

    frames = (frames .- μ) ./ (threshold * σ)
    map!(x -> x > 0 ? x : 0, frames, frames)
    map!(floor, frames, frames)
    if make_movie
        result = permutedims(cumsum(frames, dims=3), (2, 1, 3))
    else
        result = dropdims(sum(frames, dims=3), dims=3)'
    end
end

function remove_noise!(counts, xd, yd, xc, yc, threshold)
    @tullio counts[i, j, 1, l] *= xd[i-1]^2 + yd[j-1]^2 > threshold ? 0 : 1
    @tullio counts[i, j, 2, l] *= xc[i-1]^2 + yc[j-1]^2 > threshold ? 0 : 1
    @tullio counts[1, j, k, l] = 0
    @tullio counts[0, 1, k, l] = 0
    return nothing
end

function history_vector(counts)
    history = Int64[]
    for n ∈ 2:size(counts, 4)
        Δ = counts[:, :, :, n] - counts[:, :, :, n-1]
        for k ∈ eachindex(Δ)
            for _ ∈ 1:Δ[k]
                push!(history, k)
            end
        end
    end
    history
end

function get_n_detections(n, history)
    @assert n <= length(history) "There are only $(length(history)) detections"

    outcomes = Dict{Int,Int}()
    for event ∈ history[1:n]
        outcomes[event] = get(outcomes, event, 0) + 1
    end
    outcomes
end

function get_n_detections(n, frames, background, threshold)
    counts = extract_counts(frames, background, threshold; make_movie=false)
    history = history_vector(counts)
    get_n_detections(n, history)
end

function fit_grid(image)
    function f(extrema)
        xmin = extrema[1]
        ymin = extrema[2]
        xmax = extrema[3]
        ymax = extrema[4]
        x = LinRange(xmin, xmax, size(image, 1))
        y = LinRange(ymin, ymax, size(image, 2))

        @tullio prediction[i, j] := exp(-x[i]^2 - y[j]^2)
        mapreduce((x, y) -> (x - y)^2, +, normalize(image), normalize(prediction))
    end
    optimize(f, [-4.0, -4.0, 4.0, 4.0])
end
##
_calibration = get_value.(TiffImages.load("ExperimentalData/UFMG/calibration.tif"))
calibration = stack([_calibration'[65:end, :], _calibration'[1:64, :]])

dresult = fit_grid(calibration[:, :, 1])
cresult = fit_grid(calibration[:, :, 2])
xd = LinRange(dresult.minimizer[1], dresult.minimizer[3], 65)
yd = LinRange(dresult.minimizer[2], dresult.minimizer[4], 65)
xc = LinRange(cresult.minimizer[1], cresult.minimizer[3], 65)
yc = LinRange(cresult.minimizer[2], cresult.minimizer[4], 65)

order = 1
direct_operators = assemble_position_operators(xd, yd, order)

mode_converter = diagm([(-im)^k for k ∈ 0:order])
astig_operators = assemble_position_operators(xc, yc, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
mthd = MetropolisHastings(; nchains=8)

background = get_value.(TiffImages.load("ExperimentalData/UFMG/background.tif"))
##
frames = get_value.(TiffImages.load("ExperimentalData/UFMG/order1/raw/a5.tif"))

_counts = extract_counts(frames, background, 5, make_movie=true);
counts = Int.(stack([_counts[65:end, :, :], _counts[1:64, :, :]], dims=3))
remove_noise!(counts, xd, yd, xc, yc, 5)
file = h5open("ExperimentalData/UFMG/results.h5", "r")
#history = history_vector(counts)
history = read(file["6"])
close(file)
##
outcomes = get_n_detections(500, history)

img = array_representation(outcomes, (64, 64, 2))
heatmap(vcat(img[:, :, 1], img[:, :, 2]), colormap=:hot)
##
order = 1
file = h5open("ExperimentalData/UFMG/coefficients.hdf5")
coeffs = file["order$order"][:, 6]
close(file)
nobs = 2 .^ (1:9)
fids = Vector{Float64}(undef, length(nobs))

_counts = extract_counts(frames, background, 5, make_movie=true)
counts = Int.(stack([_counts[65:end, :, :], _counts[1:64, :, :]], dims=3))
remove_noise!(counts, xd, yd, xc, yc, 5)
history = history_vector(counts)

for (k, n) ∈ enumerate(nobs)
    outcomes = get_n_detections(n, history)
    pred_angles = prediction(outcomes, operators, mthd)
    ψ_pred = hurwitz_parametrization(pred_angles)
    fids[k] = abs2(ψ_pred ⋅ coeffs)
end
fids
##
fig = Figure()
ax = Axis(fig[1, 1],
    xscale=log2,
    yscale=log10,
    xticks=[2^n for n in 1:13],
    xlabel="Number of observations",
    ylabel="Infidelity")
lines!(ax, nobs, 1 .- fids)
fig