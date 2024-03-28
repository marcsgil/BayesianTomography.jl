"""
    HaarUnitary(dim::Int)

A type representing a Haar-random unitary matrix of dimension `dim`.
"""
struct HaarUnitary
    dim::Int
end

"""
    sample(type, n_samples)
    sample(type)

Sample `n_samples` from `type`.

If `n_samples` is not provided, a single sample is returned.

Possible values for type are [`HaarUnitary`](@ref), [`HaarVector`](@ref), [`Simplex`](@ref), [`ProductMeasure`](@ref), and [`GinibreEnsamble`](@ref).
"""
function sample(type::HaarUnitary, n_samples)
    Zs = randn(ComplexF32, type.dim, type.dim, n_samples)

    for Z ∈ eachslice(Zs, dims=3)
        Q, R = qr(Z)
        Λ = diag(R)
        @. Λ /= abs(Λ)
        Z .= Q * diagm(Λ)
    end

    return Zs
end

"""
    HaarVector(dim::Int)

A type representing a Haar-random unitary vector of dimension `dim`.
"""
struct HaarVector
    dim::Int
end

function sample(type::HaarVector, n_samples)
    sample(HaarUnitary(type.dim), n_samples)[1, :, :]
end

"""
    Simplex(dim::Int)

A type representing a random point on the simplex embeded in a space of dimension `dim`.
"""
struct Simplex
    dim::Int
end

function sample(type::Simplex)
    dim = type.dim
    ξs = rand(Float32, dim - 1)
    λs = Vector{Float32}(undef, dim)
    for (k, ξ) ∈ enumerate(ξs)
        λs[k] = (1 - ξ^(1 / (dim - k))) * (1 - sum(λs[1:k-1]))
    end
    λs[end] = 1 - sum(λs[1:end-1])
    λs
end

function sample(type::Simplex, nsamples)
    stack(sample(type) for _ ∈ 1:nsamples)
end

function combine!(unitaries::Array{T1,3}, probabilities::Array{T2,2}) where {T1,T2}
    @assert size(probabilities, 2) == size(unitaries, 3) "The number of probabilities and unitaries must be the same."
    @assert size(probabilities, 1) == size(unitaries, 1) "The dimension of the probabilities and unitaries must be the same."

    for (U, p) ∈ zip(eachslice(unitaries, dims=3), eachslice(probabilities, dims=2))
        U .= U * diagm(p) * U'
    end
end

"""
    ProductMeasure(dim::Int)

A type representing a measure on the density states.
It is a product Haar measure on the unitary group and a uniform (Lebesgue) measure on the simplex.
"""
struct ProductMeasure
    dim::Int
end

function sample(type::ProductMeasure, n_samples)
    dim = type.dim
    ps = sample(Simplex(dim), n_samples)
    Us = sample(HaarUnitary(dim), n_samples)
    combine!(Us, ps)
    Us
end

"""
    GinibreEnsamble(dim::Int)

A type representing a Ginibre ensamble of complex matrices of dimension `dim`.
"""
struct GinibreEnsamble
    dim::Int
end

function sample(type::GinibreEnsamble, n_samples)
    ρs = randn(ComplexF32, type.dim, type.dim, n_samples)

    for ρ ∈ eachslice(ρs, dims=3)
        ρ .= ρ * ρ'
        ρ ./= tr(ρ)
    end

    ρs
end

function sample(type)
    s = sample(type, 1)
    dropdims(s, dims=ndims(s))
end