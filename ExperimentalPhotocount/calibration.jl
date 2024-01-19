using TiffImages, Statistics, HDF5, Tullio, Optim, LinearAlgebra

get_value(x) = x.val

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

function write_calibration_info(dest, src)
    _calibration = get_value.(TiffImages.load(src))
    L = size(_calibration, 1)
    calibration = stack([_calibration'[L+1:end, :], _calibration'[1:L, :]])

    direct_limits = fit_grid(calibration[:, :, 1]).minimizer
    converted_limits = fit_grid(calibration[:, :, 2]).minimizer

    file = h5open(dest, "cw")
    file["direct_limits"] = direct_limits
    file["converted_limits"] = converted_limits
    close(file)
end

order = 1

write_calibration_info("ExperimentalData/UFMG/Order$order/results.h5",
    "ExperimentalData/UFMG/Order$order/calibration.tif")