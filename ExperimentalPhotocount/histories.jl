using TiffImages, HDF5, Statistics, Tullio, ProgressMeter

get_value(x) = x.val

function extract_counts(frames, μ, σ, threshold)
    frames = (frames .- μ) ./ (threshold * σ)
    map!(x -> x > 0 ? Int(floor(x)) : 0, frames, frames)
    permutedims(cumsum(frames, dims=3), (2, 1, 3))
end

function remove_noise!(counts, xd, yd, xc, yc, threshold)
    @tullio counts[i, j, 1, l] *= xd[i]^2 + yd[j]^2 > threshold ? 0 : 1
    @tullio counts[i, j, 2, l] *= xc[i]^2 + yc[j]^2 > threshold ? 0 : 1
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

function write_history(key, dest, src, bg_src, calib_src, threshold)
    bg_file = h5open(bg_src, "r")
    μ = read(bg_file["mean_background"])
    σ = read(bg_file["std_background"])
    close(bg_file)

    calib_file = h5open(calib_src, "r")
    direct_limits = calib_file["direct_limits"][:]
    converted_limits = calib_file["converted_limits"][:]
    close(calib_file)

    frames = get_value.(TiffImages.load(src))
    _counts = extract_counts(frames, μ, σ, threshold)
    L = size(_counts, 1) ÷ 2
    counts = stack([_counts[L+1:end, :, :], _counts[1:L, :, :]], dims=3)

    xd = LinRange(direct_limits[1], direct_limits[3], L)
    yd = LinRange(direct_limits[2], direct_limits[4], L)
    xc = LinRange(converted_limits[1], converted_limits[3], L)
    yc = LinRange(converted_limits[2], converted_limits[4], L)
    remove_noise!(counts, xd, yd, xc, yc, 5)

    file = h5open(dest, "cw")
    file[key] = history_vector(counts)
    close(file)
end

order = 1

@showprogress for i ∈ 0:49
    write_history("$(i+1)", "ExperimentalData/UFMG/Order$order/results.h5",
        "ExperimentalData/UFMG/Order$order/$i.tif",
        "ExperimentalData/UFMG/Order$order/results.h5",
        "ExperimentalData/UFMG/Order$order/results.h5", 5)
end