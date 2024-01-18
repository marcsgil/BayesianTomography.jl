using TiffImages, CairoMakie, Statistics

get_value(x) = x.val

backgorund = get_value.(TiffImages.load("tiffs/bg_256x8ns_4x4.tif"))
frames = get_value.(TiffImages.load("tiffs/hg10_256x8ns_4x4.tif"))
##
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

counts = extract_counts(frames, backgorund, 5, make_movie=true)
##
partial_counts = counts[:, :, 5000]

fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], aspect=2)
heatmap!(ax, partial_counts, colormap=:hot)
ax.title = string("Counts: ", Int(sum(partial_counts)))
fig
##
counts = extract_counts(frames, backgorund, 5, make_movie=true)


ncounts = LinRange(1, size(counts, 3), size(counts, 3) ÷ 25)
ncounts = round.(Int, ncounts)

fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1], title=string("Counts: ", Int(sum(counts))), aspect=2)
hm = heatmap!(ax, counts[:, :, 1], colormap=:hot)

record(fig, "movie.mp4", ncounts) do ncounts
    ax.title = string("Counts: ", Int(sum(counts[:, :, ncounts])))
    heatmap!(hm, counts[:, :, ncounts], colormap=:hot)
end