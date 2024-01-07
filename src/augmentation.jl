function compose_povm(args...; probabilities=fill(1 / length(args), length(args)))
    stack(probabilities[n] * arg for (n, arg) in enumerate(args))
end

function unitary_transform(operators, unitary)
    [unitary' * operator * unitary for operator in operators]
end

function augment_povm(povm, unitaries...; probabilities=fill(1 / (length(unitaries) + 1), length(unitaries) + 1))
    compose_povm(povm, (unitary_transform(povm, unitary) for unitary ∈ unitaries)...; probabilities)
end