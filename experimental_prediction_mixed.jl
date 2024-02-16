using BayesianTomography, HDF5, Tullio, CairoMakie, Optim, LinearAlgebra, Images, ProgressMeter

function build_basis(xd, yd, xc, yc, order, angle)
    dbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(xd, yd))
        for n ∈ 0:order])
    cbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)) * cis(-angle * n), Iterators.product(xc, yc))
        for n ∈ 0:order])
    stack([dbasis, cbasis], dims=3)
end

function build_basis(x, y, order, angle)
    dbasis = stack([
        map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(x, y))
        for n ∈ 0:order])
    @tullio cbasis[x, y, i] := dbasis[x, y, i] * cis(-angle * n)
    stack([dbasis, cbasis], dims=3)
end

function fit_grid(image, coeffs, angle)
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

        for (n, element) ∈ enumerate(eachslice(basis, dims=3))
            element .*= cis(-angle * n)
        end

        @tullio prediction[x, y] := coeffs[k, j] * basis[x, y, j] * conj(basis[x, y, k]) |> real
        mapreduce((x, y) -> (x - y)^2, +, normalize(image), normalize(prediction))
    end
    optimize(f, [-4.0, -4.0, 4.0, 4.0])
end

function fit_basis(image, coeffs, angle)
    dresult = fit_grid(image[:, :, 1], coeffs, 0)
    cresult = fit_grid(image[:, :, 2], coeffs, angle)
    xd = LinRange(dresult.minimizer[1], dresult.minimizer[3], size(image, 1))
    yd = LinRange(dresult.minimizer[2], dresult.minimizer[4], size(image, 2))
    xc = LinRange(cresult.minimizer[1], cresult.minimizer[3], size(image, 1))
    yc = LinRange(cresult.minimizer[2], cresult.minimizer[4], size(image, 2))
    build_basis(xd, yd, xc, yc, size(coeffs, 1) - 1, angle)
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
file = h5open("ExperimentalData/mixed_dataset.h5")
_images = read(file["images_order$order"])
images = stack(imresize(image, 64, 64) for image ∈ eachslice(_images, dims=(3, 4)))
ρs = read(file["labels_order$order"])
close(file)
##
r = LinRange(-3, 3, 512)
basis = fit_basis(images[:, :, :, 1], ρs[:, :, 1], π / 6)
@tullio timage[x, y, i, image] := ρs[k, j, image] * basis[x, y, i, j] * conj(basis[x, y, i, k]) |> real
##
index = 6
astig = 2

fig = Figure(resolution=(1000, 500))
ax1 = CairoMakie.Axis(fig[1, 1],
    aspect=1)
ax2 = CairoMakie.Axis(fig[1, 2],
    aspect=1)
heatmap!(ax1, images[:, :, astig, index])
heatmap!(ax2, timage[:, :, astig, index])
fig
##
index = 1

dresult = fit_grid(images[:, :, 1, index], ρs[:, :, 1], 0)
cresult = fit_grid(images[:, :, 2, index], ρs[:, :, 1], π / 6)
xd = LinRange(dresult.minimizer[1], dresult.minimizer[3], 64)
yd = LinRange(dresult.minimizer[2], dresult.minimizer[4], 64)
xc = LinRange(cresult.minimizer[1], cresult.minimizer[3], 64)
yc = LinRange(cresult.minimizer[2], cresult.minimizer[4], 64)

direct_operators = assemble_position_operators(xd, yd, order)

mode_converter = diagm([cis(k * π / 6) for k ∈ 0:order])
astig_operators = assemble_position_operators(xc, yc, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
##
fids = Vector{Float64}(undef, 100)
ispossdef = Vector{Bool}(undef, 100)
mthd = LinearInversion(operators)

function is_positive_semi_definite(A)
    # Compute the eigenvalues of A
    eigenvalues = eigvals(A)

    # Check if all eigenvalues are non-negative
    return all(real(eigenvalue) >= 0 for eigenvalue in eigenvalues)
end

p = Progress(length(fids));
for n ∈ eachindex(fids)
    treated_image = treat_image(images[:, :, :, n])
    probs = normalize(treated_image, 1)
    #pred_angles = prediction(outcomes, operators, mthd)
    σ = prediction(probs, mthd)
    #fids[n] = abs2(coeffs[:, n] ⋅ hurwitz_parametrization(pred_angles))
    sqrt_ρ = sqrt(ρs[:, :, n])
    fids[n] = abs2(tr(sqrt((sqrt_ρ * σ * sqrt_ρ))))
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