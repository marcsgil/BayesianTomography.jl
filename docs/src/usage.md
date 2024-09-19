# Basic Usage

Possibly the simplest setup for quantum state tomography is the tomography of the polarization state of a single photon, as illustrated bellow.

![Polarization Setup](assets/polarization_setup.jpeg)

The following code snippet demonstrates how this can be modeled in the package:

```@example
import Random #hide
Random.seed!(1234) #hide
using BayesianTomography

#Define the POVM for the polarization tomography
symbols = [:H, :V, :D, :A, :R, :L]
measurements = [get_projector(polarization_state(Val(s))) for s in symbols]

problem = StateTomographyProblem(measurements)

#Linear inversion method
mthd = LinearInversion(problem)

#Generate a random quantum state to be used as an example.
ρ = sample(ProductMeasure(2))

#Simulate outcomes
#Note that we need a large number of outcomes for this method to work well.
outcomes = simulate_outcomes(ρ, measurements, 10^6) 
σ = prediction(outcomes, mthd)[1] #Make a prediction
fidelity(ρ, σ) #Calculate the fidelity
```