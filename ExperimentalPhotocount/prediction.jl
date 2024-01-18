using HDF5, BayesianTomography, LinearAlgebra, ProgressMeter, CairoMakie

function get_n_detections(n, history)
    @assert n <= length(history) "There are only $(length(history)) detections"

    outcomes = Dict{Int,Int}()
    for event ∈ history[1:n]
        outcomes[event] = get(outcomes, event, 0) + 1
    end
    outcomes
end

calib_file = h5open("ExperimentalData/UFMG/Order1/results.h5", "r")
direct_limits = calib_file["direct_limits"][:]
converted_limits = calib_file["converted_limits"][:]
close(calib_file)

xd = LinRange(direct_limits[1], direct_limits[3], 65)
yd = LinRange(direct_limits[2], direct_limits[4], 65)
xc = LinRange(converted_limits[1], converted_limits[3], 65)
yc = LinRange(converted_limits[2], converted_limits[4], 65)

order = 1
direct_operators = assemble_position_operators(xd, yd, order)

mode_converter = diagm([(-im)^k for k ∈ 0:order])
astig_operators = assemble_position_operators(xc, yc, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
mthd = MetropolisHastings(; nchains=10)
##
order = 1
file = h5open("ExperimentalData/UFMG/coefficients.hdf5")
coeffs = read(file["order$order"])
close(file)
nobs = map(x -> round(Int, 2.0^x), LinRange(5, 11, 10))
idxs = 1:50
fids = Vector(undef, length(nobs))

file = h5open("ExperimentalData/UFMG/Order1/results.h5")
@showprogress for (i, nobs) ∈ enumerate(nobs)
    _fids = Float64[]
    for j ∈ idxs
        history = read(file["$j"])
        try
            outcomes = get_n_detections(nobs, history)
            pred_angles = prediction(outcomes, operators, mthd)
            ψ_pred = hurwitz_parametrization(pred_angles)
            push!(_fids, abs2(ψ_pred ⋅ coeffs[:, j]))
        catch
            continue
        end
    end
    fids[i] = mean(_fids)
end
close(file)
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
file = h5open("ExperimentalData/UFMG/Order1/results.h5")
nphotons = [length(read(file["$j"])) for j ∈ 1:50]
close(file)

hist(nphotons, bins=10)


mean(nphotons)
count(x -> x < 1000, nphotons)