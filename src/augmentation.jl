function compose_povm(povms...; probabilities=fill(Float32(1 / length(povms)), length(povms)))
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

function augment_povm(povm, unitaries...; probabilities=fill(1 / (length(unitaries) + 1), length(unitaries) + 1))
    compose_povm(povm, (unitary_transform(povm, unitary) for unitary ∈ unitaries)...; probabilities)
end