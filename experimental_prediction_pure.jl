using BayesianTomography, HDF5, Tullio, CairoMakie, Optim, LinearAlgebra, Images, ProgressMeter

function build_basis(xd, yd, xc, yc, order)
    dbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(xd, yd))
        for n ∈ 0:order])
    cbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)) * (-im)^n, Iterators.product(xc, yc))
        for n ∈ 0:order])
    stack([dbasis, cbasis], dims=3)
end

function build_basis(x, y, order)
    dbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(x, y))
        for n ∈ 0:order])
    @tullio cbasis[x, y, i] := dbasis[x, y, i] * (-im)^i
    stack([dbasis, cbasis], dims=3)
end

function fit_grid(image, coeffs, is_astig)
    function f(extrema)
        xmin = extrema[1]
        ymin = extrema[2]
        xmax = extrema[3]
        ymax = extrema[4]
        x = LinRange(xmin, xmax, size(image, 1))
        y = LinRange(ymin, ymax, size(image, 2))
        basis = stack([
            map(r -> complex(hg([r[1], r[2]], (order - n, n))), Iterators.product(x, y))
            for n ∈ 0:order])

        if is_astig
            @tullio basis[x, y, i] *= (-im)^i
        end

        @tullio prediction[x, y] := coeffs[j] * basis[x, y, j] |> abs2
        mapreduce((x, y) -> (x - y)^2, +, normalize(image), normalize(prediction))
    end
    optimize(f, [-4.0, -4.0, 4.0, 4.0])
end

function fit_basis(image, coeffs)
    dresult = fit_grid(image[:, :, 1], coeffs, false)
    cresult = fit_grid(image[:, :, 2], coeffs, true)
    xd = LinRange(dresult.minimizer[1], dresult.minimizer[3], size(image, 1))
    yd = LinRange(dresult.minimizer[2], dresult.minimizer[4], size(image, 2))
    xc = LinRange(cresult.minimizer[1], cresult.minimizer[3], size(image, 1))
    yc = LinRange(cresult.minimizer[2], cresult.minimizer[4], size(image, 2))
    build_basis(xd, yd, xc, yc, length(coeffs) - 1)
end

function treat_image(image; res=nothing, counts=nothing)
    background = minimum(image) + 2
    _image = similar(image)
    for n ∈ eachindex(_image)
        _image[n] = image[n] > background ? image[n] - background : zero(eltype(image))
    end

    if !isnothing(res)
        _image = imresize(_image, res, res)
    end

    if isnothing(counts)
        return round.(Int, _image)
    else
        return round.(Int, _image * counts / sum(_image))
    end
end

order = 1
file = h5open("datasets/pure_dataset.h5")
_images = read(file["images_order$order"])
images = stack(imresize(image, 64, 64) for image ∈ eachslice(_images, dims=(3, 4)))
_coeffs = read(file["coefficients_order$order"])
coeffs = _coeffs[1:order+1, :] + im * _coeffs[order+2:end, :]
close(file)
##
r = LinRange(-3, 3, 512)
basis = fit_basis(images[:, :, :, 1], coeffs[:, 1])
@tullio timage[x, y, i, image] := coeffs[j, image] * basis[x, y, i, j] |> abs2
##
index = 1
astig = 1

fig = Figure(resolution=(1000, 500))
ax1 = CairoMakie.Axis(fig[1, 1],
    aspect=1)
ax2 = CairoMakie.Axis(fig[1, 2],
    aspect=1)
heatmap!(ax1, images[:, :, astig, index])
heatmap!(ax2, timage[:, :, astig, index])
fig
##
index = 2

dresult = fit_grid(images[:, :, 1, index], coeffs[:, 1], false)
cresult = fit_grid(images[:, :, 2, index], coeffs[:, 1], true)
xd = LinRange(dresult.minimizer[1], dresult.minimizer[3], 65)
yd = LinRange(dresult.minimizer[2], dresult.minimizer[4], 65)
xc = LinRange(cresult.minimizer[1], cresult.minimizer[3], 65)
yc = LinRange(cresult.minimizer[2], cresult.minimizer[4], 65)

direct_operators = assemble_position_operators(xd, yd, order)

mode_converter = diagm([(-im)^k for k ∈ 0:order])
astig_operators = assemble_position_operators(xc, yc, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
##
fids = Vector{Float64}(undef, 100)
ispossdef = Vector{Bool}(undef, 100)
#mthd = MetropolisHastings(; nchains=8)
mthd = LinearInversion(order + 1; α=3)

function is_positive_semi_definite(A)
    # Compute the eigenvalues of A
    eigenvalues = eigvals(A)

    # Check if all eigenvalues are non-negative
    return all(real(eigenvalue) >= 0 for eigenvalue in eigenvalues)
end

p = Progress(length(fids))
for n ∈ eachindex(fids)
    treated_image = treat_image(images[:, :, :, n])
    outcomes = dict_representation(treated_image)
    #pred_angles = prediction(outcomes, operators, mthd)
    ρ = prediction(normalize(treated_image, 1), operators, mthd)
    #fids[n] = abs2(coeffs[:, n] ⋅ hurwitz_parametrization(pred_angles))
    fids[n] = real(dot(coeffs[:, n], ρ, coeffs[:, n]))
    #ispossdef[n] = is_positive_semi_definite(ρ)
    next!(p)
end
finish!(p)

mean(fids)
##
file = h5open("results/exp_fids.h5", "cw")
file["order$order"] = mean(fids)
close(file)
##
file = h5open("results/exp_fids.h5", "r")
fids = [read(file["order$order"]) for order ∈ 1:5]
close(file)

fig = Figure()
ax = CairoMakie.Axis(fig[1, 1],
    xlabel="Order",
    ylabel="Mean Fidelity",
    title="Experimental Fidelities")
ylims!(ax, 0.96, 1.001)
lines!(1:5, fids)
fig