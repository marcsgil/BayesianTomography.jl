"""
    compose_povm(povms::AbstractArray{Matrix{T}}...; weights=fill(one(T) / length(povms), length(povms))) where {T}

Compose a POVM (Positive Operator-Valued Measure) from a set of given POVMs.

# Arguments
- `povms`: Variable number of POVMs. Each POVM is represented as an array of matrices.
- `weights`: An optional array of weights associated with each POVM. If not provided, it defaults to a uniform distribution.

# Returns
- A new POVM that is a composition of the input POVMs, weighted by their respective weights.

# Example
```julia
povm1 = [rand(2,2) for _ in 1:3]
povm2 = [rand(2,2) for _ in 1:3]
composed_povm = compose_povm(povm1, povm2)
```
"""
function compose_povm(povms::AbstractArray{Matrix{T}}...;
    weights=fill(one(T) / length(povms), length(povms))) where {T}
    stack(weights[n] * povm for (n, povm) in enumerate(povms))
end

"""
    unitary_transform!(povm, unitary)

Apply a unitary transformation to each element of a given POVM (Positive Operator-Valued Measure), modifing it in place.

# Arguments
- `povm`: The POVM to be transformed. It is represented as an array of matrices.
- `unitary`: The unitary matrix representing the transformation to be applied.

# Example
```julia
bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
unitary_transform!(bs_povm, half_wave_plate)
```
"""
function unitary_transform!(povm, unitary)
    for n ∈ eachindex(povm)
        povm[n] = unitary' * povm[n] * unitary
    end
end


"""
    unitary_transform(povm, unitary)

Apply a unitary transformation to each element of a given POVM (Positive Operator-Valued Measure).

# Arguments
- `povm`: The POVM to be transformed. It is represented as an array of matrices.
- `unitary`: The unitary matrix representing the transformation to be applied.

# Returns
- A new POVM that is the result of applying the unitary transformation to the input POVM.

# Example
```julia
bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
unitary_transform!(bs_povm, half_wave_plate)
```
"""
function unitary_transform(povm, unitary)
    _povm = copy(povm)
    unitary_transform!(_povm, unitary)
    _povm
end

"""
    augment_povm(povm::AbstractArray{Matrix{T}}, unitaries...; 
        weights=fill(one(T) / (length(unitaries) + 1), length(unitaries) + 1) where {T}

Augment a POVM (Positive Operator-Valued Measure) by applying a set of unitary transformations to it.

# Arguments
- `povm`: The POVM to be augmented. It is represented as an array of matrices.
- `unitaries`: Variable number of unitary matrices representing the transformations to be applied.
- `weights`: An optional array of weights associated with each unitary transformation. If not provided, it defaults to a uniform distribution.

# Returns
- A new POVM that is the result of applying the unitary transformations to the input POVM.

# Example
```julia
bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]]
half_wave_plate = [1 1; 1 -1] / √2
quater_wave_plate = [1 im; im 1] / √2
povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, weights=[1 / 2, 1 / 4, 1 / 4])
```
"""
function augment_povm(povm::AbstractArray{Matrix{T}}, unitaries...;
    weights=fill(one(T) / (length(unitaries) + 1), length(unitaries) + 1)) where {T}
    compose_povm(povm, (unitary_transform(povm, unitary) for unitary ∈ unitaries)...; weights)
end