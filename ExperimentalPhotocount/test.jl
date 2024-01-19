using HDF5, CairoMakie, BayesianTomography, Tullio

idx = 20
order = 3
file = h5open("ExperimentalData/UFMG/Order$order/results.h5", "r")
history = read(file[string(idx)])
close(file)

file = h5open("ExperimentalData/UFMG/coefficients.hdf5")
coeffs = file["order$order"][:, idx]
close(file)

image = array_representation(history, (64, 64, 2))
image = vcat(image[:, :, 1], image[:, :, 2])
heatmap(image, colormap=:hot)
##

x = LinRange(-3.5, 3.5, 64)
y = x

@tullio dbasis[i, j, k] := hg([x[i], y[j]], (order - k + 1, k - 1)) (i ∈ eachindex(x), j ∈ eachindex(y), k ∈ 1:order+1)
@tullio cbasis[i, j, k] := (-im)^k * hg([x[i], y[j]], (order - k + 1, k - 1)) (i ∈ eachindex(x), j ∈ eachindex(y), k ∈ 1:order+1)

dbasis = Array(dbasis)
cbasis = Array(cbasis)

@tullio Id[i, j] := dbasis[i, j, k] * coeffs[k] |> abs2
@tullio Ic[i, j] := cbasis[i, j, k] * coeffs[k] |> abs2

heatmap(vcat(Id, Ic), colormap=:hot)