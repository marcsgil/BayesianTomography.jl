# API

## Prediction

```@docs
prediction(::Any, ::LinearInversion)
prediction(::Any, ::BayesianInference)
LinearInversion
BayesianInference
```

## Augmentation

```@docs
compose_povm
unitary_transform!
unitary_transform
augment_povm
```

## Generalized Gell-Mann matrices

```@docs
triangular_indices
X_matrix
Y_matrix
Z_matrix
gell_mann_matrices
basis_decomposition
```

## Representations

```@docs
History
reduced_representation(::Array{T,N}) where {T,N}
reduced_representation(::History)
complete_representation(::Matrix{T}, ::Any) where {T}
complete_representation(::History{T}, ::Any) where {T}
```

## Samplers

```@docs
sample
HaarUnitary
HaarVector
Simplex
ProductMeasure
GinibreEnsamble
```

## Simulation

```@docs
simulate_outcomes
simulate_outcomes!
```

## Misc

```@docs
fidelity
project2density
project2pure
orthogonal_projection
real_orthogonal_projection
linear_combination
linear_combination!
isposdef!(::Any,::Any,::Any)
cond
maximally_mixed_state
```
