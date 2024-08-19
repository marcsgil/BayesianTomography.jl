"""
    LinearInversion(povm)

Construct a linear inversion method for quantum state tomography.
"""
struct LinearInversion{T1<:Real,T2<:Union{Real,Complex}}
    povm::Matrix{T1}
    basis::Array{T2,3}
    dim::Int
    pseudo_inv::Matrix{T1}
end

function LinearInversion(povm::Matrix{T}, basis=gell_mann_matrices(Int(√size(povm, 2)), complex(T))) where {T<:Real}
    dim = Int(√size(povm, 2))
    pseudo_inv = pinv(@view povm[:, begin+1:end])
    LinearInversion(povm, basis, dim, pseudo_inv)
end

function LinearInversion(povm::Array{Matrix{T}}, basis=gell_mann_matrices(size(first(povm), 1), complex(T))) where {T}
    dim = size(first(povm), 1)
    povm_matrix = stack(Π -> real_orthogonal_projection(Π, basis), vec(povm), dims=1)
    pseudo_inv = pinv(@view povm_matrix[:, begin+1:end])
    LinearInversion(povm_matrix, basis, dim, pseudo_inv)
end

function get_probs(mthd, θ)
    dest = Vector{eltype(θ)}(undef, size(mthd.povm, 1))
    get_probs!(dest, mthd, θ)
    dest
end

function get_probs!(dest::Vector{T}, mthd, θ) where {T}
    copy!(dest, (@view mthd.povm[:, begin]))
    mul!(dest, (@view mthd.povm[:, begin+1:end]), θ, one(T), convert(T, mthd.dim^(-1 // 2)))
end

"""
    prediction(outcomes, method::LinearInversion)

Predict the quantum state from the outcomes of a tomography experiment using the [`LinearInversion`](@ref) method.
"""
function prediction(outcomes::Vector{T}, method::LinearInversion{T1,T2}) where {T<:AbstractFloat,T1,T2}
    θs = method.pseudo_inv * outcomes
    mul!(θs, method.pseudo_inv, view(method.povm, :, 1), convert(T1, -1 / √method.dim), one(T1))

    linear_combination(vcat(convert(T1, 1 / √method.dim), θs), method.basis), θs, covariance(outcomes, method, θs)
end

function prediction(outcomes::Vector{T}, method::LinearInversion) where {T<:Integer}
    prediction(normalize(outcomes, 1), method)
end

function prediction(outcomes::AbstractArray{T}, method::LinearInversion{T1,T2}) where {T<:Real,T1,T2}
    prediction(vec(outcomes), method)
end

function covariance(outcomes, method::LinearInversion{T1,T2}, θs) where {T1,T2}
    """N = sum(outcomes)
    povm = method.povm

    sum_residues = zero(eltype(povm))

    @inbounds for i in axes(povm, 1)
        temp = zero(sum_residues)

        temp += outcomes[i] / N - method.correction[i]
        @simd for j in axes(povm, 2)
            temp -= povm[i, j] * θs[j]
        end

        sum_residues += abs2(temp)
    end

    inv(povm' * povm) * sum_residues / (size(povm, 1) - size(povm, 2))"""
    rand(T1, method.dim - 1, method.dim - 1)
end