function compose_povm(povms::AbstractArray{Matrix{T}}...;
    probabilities=fill(one(T) / length(povms), length(povms))) where {T}
    stack(probabilities[n] * povm for (n, povm) in enumerate(povms))
end

function unitary_transform!(povm, unitary)
    for n ∈ eachindex(povm)
        povm[n] = unitary' * povm[n] * unitary
    end
end

function unitary_transform(povm, unitary)
    _povm = copy(povm)
    unitary_transform!(_povm, unitary)
    _povm
end

function augment_povm(povm::AbstractArray{Matrix{T}}, unitaries...;
    probabilities=fill(one(T) / (length(unitaries) + 1), length(unitaries) + 1)) where {T}
    compose_povm(povm, (unitary_transform(povm, unitary) for unitary ∈ unitaries)...; probabilities)
end