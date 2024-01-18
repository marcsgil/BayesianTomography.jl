using TiffImages, CairoMakie, Statistics, BayesianTomography, Optim, Tullio, LinearAlgebra

get_value(x) = x.val

function extract_counts(frames, backgorund, threshold; make_movie=false)
    μ = dropdims(mean(backgorund, dims=3), dims=3)
    σ = dropdims(std(backgorund, dims=3), dims=3)

    frames = (frames .- μ) ./ (threshold * σ)
    map!(x -> x > 0 ? x : 0, frames, frames)
    map!(floor, frames, frames)
    if make_movie
        result = permutedims(cumsum(frames, dims=3), (2, 1, 3))
    else
        result = dropdims(sum(frames, dims=3), dims=3)'
    end
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

_calibration = get_value.(TiffImages.load("luz_apagada/gaussian_256x8ns_4x4.tif"))
calibration = stack([_calibration'[1:64, :], _calibration'[65:end, :]])

dresult = fit_grid(calibration[:, :, 2])
cresult = fit_grid(calibration[:, :, 1])
xd = LinRange(dresult.minimizer[1], dresult.minimizer[3], 65)
yd = LinRange(dresult.minimizer[2], dresult.minimizer[4], 65)
xc = LinRange(cresult.minimizer[1], cresult.minimizer[3], 65)
yc = LinRange(cresult.minimizer[2], cresult.minimizer[4], 65)
##
##
backgorund = get_value.(TiffImages.load("luz_apagada/bg_256x8ns_4x4.tif"))
frames = get_value.(TiffImages.load("luz_apagada/hg01_256x8ns_4x4.tif"))

counts = extract_counts(frames, backgorund, 5, make_movie=true)

nimages = 1000
image = Int.(stack([counts[1:64, :, nimages], counts[65:end, :, nimages]]))
@tullio image[i, j, 1] *= xc[i-1]^2 + yc[j-1]^2 > 5 ? 0 : 1
heatmap(image[:, :, 1])
@tullio image[i, j, 2] *= xd[i-1]^2 + yd[j-1]^2 > 5 ? 0 : 1
heatmap(image[:, :, 2])
##
order = 1
direct_operators = assemble_position_operators(xd, yd, order)

mode_converter = diagm([(-im)^k for k ∈ 0:order])
astig_operators = assemble_position_operators(xc, yc, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(astig_operators, direct_operators)
mthd = MetropolisHastings(; nchains=8)

outcomes = dict_representation(image)
sum(values(outcomes))
##
pred_angles = prediction(outcomes, operators, mthd)
ψ_pred = hurwitz_parametrization(pred_angles)
abs2(ψ_pred ⋅ [0, 1])