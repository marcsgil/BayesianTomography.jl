using TiffImages, Statistics, HDF5

get_value(x) = x.val

function write_background_info(dest, src)
    background = get_value.(TiffImages.load(src))
    μ = dropdims(mean(background, dims=3), dims=3)
    σ = dropdims(std(background, dims=3), dims=3)
    file = h5open(dest, "cw")
    file["mean_background"] = μ
    file["std_background"] = σ
    close(file)
end

write_background_info("ExperimentalData/UFMG/Order1/results.h5",
    "ExperimentalData/UFMG/Order1/background.tif")