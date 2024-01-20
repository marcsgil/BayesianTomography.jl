using TiffImages, HDF5, Statistics, Tullio, ProgressMeter

get_value(x) = x.val

floor_relu(x) = x > 0 ? UInt8(floor(x)) : 0

function extract_photons(frames, μ, σ, threshold)
    @tullio treated_frames[i, j, k] := floor_relu((frames[i, j, k] - μ[i, j]) / (threshold * σ[i, j]))
end

function circular_mask!(images, radius, xd, yd, xc, yc)
    @tullio images[i, j, 1, k] *= xd[i]^2 + yd[j]^2 > radius ? 0 : 1
    @tullio images[i, j, 2, k] *= xc[i]^2 + yc[j]^2 > radius ? 0 : 1
    return nothing
end

function format(images)
    result = reshape(images, (size(images, 1), size(images, 2) ÷ 2, 2, :))
    reverse!(result, dims=3)
    permutedims(result, (2, 1, 3, 4))
end

function history_vector(images)
    history = Int64[]
    for image ∈ eachslice(images, dims=4)
        for k ∈ eachindex(image)
            for _ ∈ 1:image[k]
                push!(history, k)
            end
        end
    end
    history
end

function history_vector(srcs, config, threshold, radius)
    result = Vector{Vector{Int64}}(undef, length(srcs))
    config_file = h5open(config, "r")
    μ = read(config_file["mean_background"])
    σ = read(config_file["std_background"])
    direct_limits = config_file["direct_limits"][:]
    converted_limits = config_file["converted_limits"][:]
    xd = LinRange(direct_limits[1], direct_limits[3], 64)
    yd = LinRange(direct_limits[2], direct_limits[4], 64)
    xc = LinRange(converted_limits[1], converted_limits[3], 64)
    yc = LinRange(converted_limits[2], converted_limits[4], 64)
    close(config_file)

    Threads.@threads for i ∈ eachindex(srcs)
        src = srcs[i]
        frames = get_value.(TiffImages.load(src, verbose=false))
        photons = extract_photons(frames, μ, σ, threshold) |> format
        circular_mask!(photons, radius, yd, xd, yc, xc)
        result[i] = history_vector(photons)
    end
    result
end