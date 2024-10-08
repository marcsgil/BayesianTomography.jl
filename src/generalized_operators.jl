struct Projector{T<:AbstractVector}
    ψ::T
end

function Base.getindex(p::Projector, I::Vararg{Int,2})
    conj(p.ψ[I[1]]) * p.ψ[I[2]]
end

Base.size(p::Projector) = (length(p.ψ), length(p.ψ))
Base.size(p::Projector, i::Int) = length(p.ψ)
LinearAlgebra.tr(p::Projector) = sum(abs2, p.ψ)

GeneralizedOperator = Union{AbstractMatrix,Projector}

as_operator(x::GeneralizedOperator) = identity(x)
as_operator(x::AbstractVector) = Projector(x)
as_operator(::T) where {T} = throw(ArgumentError("`as_operator` is not implemented for type $T"))