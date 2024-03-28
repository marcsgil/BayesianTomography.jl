# BayesianTomography.jl

This package provides tools to perform quantum state tomography. It is designed to be flexible and easy to use, with a focus on providing a simple interface for common tasks. As its name suggests, its flagship feature is the ability to perform Bayesian quantum state tomography, but it also provides an interface for the simple linear inversion method.

## Installation

The package is registered in the Julia General registry and can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run `add BayesianTomography`.

## Usage

Possibly the simplest setup for quantum state tomography is the tomography of the polarization state of a single photon, as illustrated bellow.

![Polarization Setup](../images/polarization_setup.jpeg)

The following code snippet demonstrates how this can be modeled in the package:

```julia
using BaeyesianTomography

bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]] #POVM for a polarazing beam splitter
half_wave_plate = [1 1; 1 -1] / √2 #Unitary matrix for a half-wave plate
quarter_wave_plate = [1 im; im 1] / √2 #Unitary matrix for a quarter-wave plate

"""Augment the bs_povm with the action of half-wave plate and the quarter-wave plate. This is done because a single PBS is not enough to measure the polarization state of a photon."""
povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, 
                        weights=[1 / 2, 1 / 4, 1 / 4])

#Generate a random quantum state to be used as an example.
ρ = sample(GinibreEnsamble(2))

#Linear inversion method
li = LinearInversion(povm)

#Simulate outcomes
#Note that we need a large number of outcomes for this method to work well.
outcomes = simulate_outcomes(ρ, povm, 10^6) 
σ = prediction(outcomes, li) #Make a prediction
fidelity(ρ, σ) #Calculate the fidelity

#Bayesian inference method
bi = BayesianInference(povm)

#We can use a smaller number of outcomes for this method.
outcomes = simulate_outcomes(ρ, povm, 10^3) 
σ = prediction(outcomes, li) #Make a prediction
fidelity(ρ, σ) 
```

Let us break down the code snippet. First, one need to specify the measurement that is being performed. We do that by specifying a Postive Operator Value Measure (POVM), which is a collection of positive semi-definite matrices that sum to the identity. Each matrix $F$ corresponds to a measurement outcome in such a way that the probability of obtaining a given outcome is given by the Born rule $\text{Tr} \rho F$, where $\rho$ is the quantum state. In this package, any kind of collection of matrices can be used as a POVM, as long as they satisfy the POVM condition.

An example of a POVM is the one performed by a polarizing beam splitter (PBS):
```julia
bs_povm = [[1.0+im 0; 0 0], [0 0; 0 1]] #POVM for a polarazing beam splitter
```
This POVM has two outcomes, corresponding to the two possible polarizations of the photon. The first matrix corresponds to the horizontal polarization, and the second to the vertical polarization. Nonetheless, this POVM is not enough to determine the polarization state of a photon. It is called informationally incomplete. To do that, we need to add the action of a half-wave plate and a quarter-wave plate to the POVM. This is done by the `augment_povm` function:
```julia
half_wave_plate = [1 1; 1 -1] / √2 #Unitary matrix for a half-wave plate
quarter_wave_plate = [1 im; im 1] / √2 #Unitary matrix for a quarter-wave plate

"""Augment the bs_povm with the action of half-wave plate and the quarter-wave plate. This is done because a single PBS is not enough to measure the polarization state of a photon."""
povm = augment_povm(bs_povm, half_wave_plate, quater_wave_plate, 
                        weights=[1 / 2, 1 / 4, 1 / 4])
```
The half-wave and quarter-wave plates are represented by an unitary, and the POVM is augmented by the action $F\mapsto U^\dagger F U$ of these unitaries. The `weights` argument specifies the weight given for each POVM. In this case, the photons going to PBS1 only pass in through a single BS, which corresponds to a probability of $1/2$. The photons going to PBS2 and PBS3 pass in two BSs, which corresponds to a probability of $1/4$ for each.

Now, we can generate a random quantum state to be used as an example:
```julia
ρ = sample(GinibreEnsamble(2))
```

The next step is to choose a method to perform the tomography. The package provides two methods: the linear inversion method and the Bayesian inference method. The linear inversion method is the simplest and fastest method, but it assumes the knowledge of the probability of every experimental outcome, which can only be estimated with a large number of observations. The linear inversion method is chosen by creating a `LinearInversion` type:
```julia
li = LinearInversion(povm)
```

Now, we simulate the outcomes of the experiment:
```julia
outcomes = simulate_outcomes(ρ, povm, 10^6) 
```

Finally, we make a prediction of the quantum state using the `prediction` function, and comapre it with the true state using the `fidelity`:
```julia
σ = prediction(outcomes, li)
fidelity(ρ, σ)
```

The Bayesian inference method requires fewer observations to work well:
```julia
outcomes = simulate_outcomes(ρ, povm, 10^3) 
```
It is chosen by creating a `BayesianInference` type:
```julia
bi = BayesianInference(povm)
```

We now perform the same steps as before:
```julia
σ, _ = prediction(outcomes, li)
fidelity(ρ, σ)
```

