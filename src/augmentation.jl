function compose_povm(args...; probabilities=fill(Float32(1 / length(args)), length(args)))
    stack(probabilities[n] * arg for (n, arg) in enumerate(args))
end

function unitary_transform!(operators, unitary)
    for n ∈ eachindex(operators)
        operators[n] = unitary' * operators[n] * unitary
    end
end

function augment_povm(povm, unitaries...; probabilities=fill(1 / (length(unitaries) + 1), length(unitaries) + 1))
    compose_povm(povm, (unitary_transform(povm, unitary) for unitary ∈ unitaries)...; probabilities)
end