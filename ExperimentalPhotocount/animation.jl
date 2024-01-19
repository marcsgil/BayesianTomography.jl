using HDF5, CairoMakie, BayesianTomography

order = 1
file = h5open("ExperimentalData/UFMG/Order$order/results.h5", "r")
history = read(file["6"])
close(file)

image = array_representation(history, (64, 64, 2))
image = vcat(image[:, :, 1], image[:, :, 2])
heatmap(image, colormap=:hot)
##
file = h5open("ExperimentalData/UFMG/Order$order/results.h5")
nphotons = [length(read(file["$j"])) for j ∈ 1:50]
close(file)

hist(nphotons, bins=10)


mean(nphotons)
#count(x -> x < 2000, nphotons)
##
ncounts = round.(Int, LinRange(1, 4000, 400))

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1], title="Counts: 0", aspect=2)
hm = heatmap!(ax, image, colormap=:hot)

record(fig, "ExperimentalPhotocount/movie.mp4", ncounts) do ncounts
    ax.title = string("Counts: ", ncounts)
    image = array_representation(history[1:ncounts], (64, 64, 2))
    image = vcat(image[:, :, 1], image[:, :, 2])
    heatmap!(hm, image, colormap=:hot)
end