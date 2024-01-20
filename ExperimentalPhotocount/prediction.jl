using HDF5, BayesianTomography, LinearAlgebra, ProgressMeter, CairoMakie
include("histories.jl")

function get_n_detections(n, history)
    @assert n <= length(history) "There are only $(length(history)) detections"

    outcomes = Dict{Int,Int}()
    for event ∈ @view history[1:n]
        outcomes[event] = get(outcomes, event, 0) + 1
    end
    outcomes
end

file = h5open("ExperimentalResults/UFMG/config.h5", "r")
direct_limits = file["direct_limits"][:]
converted_limits = file["converted_limits"][:]
close(file)

xd = LinRange(direct_limits[1], direct_limits[3], 65)
yd = LinRange(direct_limits[2], direct_limits[4], 65)
xc = LinRange(converted_limits[1], converted_limits[3], 65)
yc = LinRange(converted_limits[2], converted_limits[4], 65)

order = 3
direct_operators = assemble_position_operators(xd, yd, order)

mode_converter = diagm([(-im)^k for k ∈ 0:order])
astig_operators = assemble_position_operators(xc, yc, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
mthd = MetropolisHastings(; nchains=8, nadapts=2000, nsamples=4000)
##
file = h5open("ExperimentalResults/UFMG/coefficients.hdf5")
coeffs = read(file["order$order"])
close(file)
nobs = map(x -> 2^x, 6:12)
idxs = 1:50
fids = fill(NaN, length(nobs))
srcs = ["ExperimentalResults/UFMG/Order$order/$i.tif" for i ∈ 0:49]
config = "ExperimentalResults/UFMG/config.h5"
histories = history_vector(srcs, config, 6, 6);
mean(length, histories)
##
#file = h5open("ExperimentalResults/UFMG/Order$order/results.h5")
@showprogress for i ∈ 7
    n = nobs[i]
    _fids = Float64[]
    Threads.@threads for j ∈ idxs
        history = histories[j]
        try
            outcomes = get_n_detections(n, history)
            pred_angles = prediction(outcomes, operators, mthd)
            ψ_pred = hurwitz_parametrization(pred_angles)
            push!(_fids, abs2(ψ_pred ⋅ coeffs[:, j]))
        catch e
            isa(e, AssertionError) ? continue : rethrow(e)
        end
    end
    fids[i] = mean(_fids)
end
#close(file)
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
##
file = h5open("ExperimentalResults/UFMG/predictions.h5", "cw")
file["order$order"] = fids
close(file)
##
file = h5open("ExperimentalResults/UFMG/predictions.h5", "r")
fids = stack(read(file["order$order"]) for order ∈ 1:2)
close(file)
using MathTeXEngine # required for texfont

textheme = Theme(fonts=(; regular=texfont(:text),
        bold=texfont(:bold),
        italic=texfont(:italic),
        bold_italic=texfont(:bolditalic)),
    fontsize=24,)

with_theme(textheme) do
    fig = Figure()
    ax = Axis(fig[1, 1],
        xscale=log2,
        #yscale=log10,
        xticks=[2^n for n in 1:13],
        yticks=0.9:0.01:1,
        xlabel="Photocounts",
        ylabel="Fidelity")
    series!(ax, nobs, fids',
        labels=["Order $i" for i in 1:2],
        color=[:red, :blue],
        linewidth=4)
    axislegend(ax, position=:rb)
    fig
end