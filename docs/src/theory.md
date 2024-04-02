# Theory

## POVMs

Let ``\text{Her}(\mathcal{H}) \subset \mathcal{H}`` be the set of hermitian operators acting on a Hilbert space ``\mathcal{H}``. Elements of ``\text{Her}(\mathcal{H})`` may be written as double kets: ``| F \rangle \! \rangle``. Such a space is also a *real* Hilbert space under the inner product

```math
\langle \! \langle E | F \rangle \! \rangle= \text{Tr} E^\dagger F = \text{Tr} E F
```

The state of a quantum system is represented by an element ``\rho`` of the set of positive semi-definite operators ``\text{Pos}(\mathcal{H}) \subset \text{Her}(\mathcal{H})`` such that ``\text{Tr} \rho = 1``. Observables are represented by elements ``A \in \text{Her}(\mathcal{H})`` so that its expectation value is given by ``\langle A \rangle = \langle \! \langle A | \rho \rangle \! \rangle``. A Positive Operator Valued Measure (POVM) is a set of observables ``\{F_m\} \subset \text{Pos}(\mathcal{H})`` with the property that ``\sum_m F_m = I``, where ``I`` is the identity operator. These operators model the possible outcomes of an experiment: outcome ``m`` happens with probability ``p(m) = \langle \! \langle F_m | \rho \rangle \! \rangle``. The POVM conditions ensure that ``p(m) \ge 0`` and ``\sum_m p(m) = 1``.

## Linear Inversion

The simplest method which solves the problem of quantum state tomography is linear inversion, which we describe in what follows. A POVM with ``M`` elements induces a linear map ``T: \text{Her}(\mathcal{H}) \to \mathbb{R}^M`` defined by the expression
```math
T| \Omega \rangle \! \rangle = \left(\langle \! \langle F_1 | \Omega \rangle \! \rangle,\ldots,\langle \! \langle F_M | \Omega \rangle \! \rangle\right).
```
In order to be suitable for tomography, we want that the measurement probabilities ``\mathbf{p} = T | \rho \rangle \! \rangle`` uniquely determine the state ``| \rho \rangle \! \rangle``. This is assured if the transformation ``T`` is injective. A POVM with this property is said to be informationally complete.

By choosing a basis ``\{\Omega_n\} \subset \text{Her}(\mathcal{H})``, we can specify an arbitrary state ``| \rho \rangle \! \rangle`` by a list of coefficients ``\mathbf{x} = (x_1,\ldots,x_N)`` such that ``| \rho \rangle \! \rangle = \sum_n x_n | \Omega_n \rangle \! \rangle``. Then, denoting by ``\mathbb{T}`` the matrix of ``T`` with respect to the canonical basis of ``\mathbb{R}^M`` and ``\{\Omega_m\}``, which has entries ``\mathbb{T}_{mn} = \langle \! \langle F_m | \Omega_n \rangle \! \rangle``, we have ``\mathbf{p} = \mathbb{T}\mathbf{x}``. When ``\mathbb{T}`` is injective, ``\mathbb{T}^\dagger\mathbb{T}`` is invertible, and then we can explicitly write the solution of the above equation as
```math
\mathbf{x} = (\mathbb{T}^\dagger\mathbb{T})^{-1} \mathbb{T}^\dagger \mathbf{p},
```
which, in theory, solves the tomography problem.

Note that, to apply this method, one needs a reliable estimate of the probabilities ``\mathbf{p}``. This can only be obtained by performing a large number of measurements.

## Bayesian tomography

Bayesian tomography is the application of Bayesian inference to the problem of quantum state tomography [^1]. The goal is to estimate the posterior distribution of the quantum state given the observed data. The posterior distribution is given by Bayes' theorem
```math
P(\rho | \mathcal{M}) = \frac{P(\mathcal{M} | \rho) P(\rho)}{P(\mathcal{M})},
```
where ``P(\mathcal{M} | \rho)`` is the likelihood of the observations ``\mathcal{M}`` given the state, ``P(\rho)`` is the prior distribution of the state, and ``P(\mathcal{M})`` is the evidence. In the case of quantum state tomography, the likelihood is given by the Born rule 
```math
P(\mathcal{M} | \rho) = \prod_m p(m)^{n_m}, \ \ \ p(m) = \langle \! \langle F_m | \rho \rangle \! \rangle,
```
where ``n_m`` is the number of times outcome ``m`` was observed. The prior distribution is a probability distribution over the space of states, which encodes any prior knowledge about the state. The evidence is the normalization constant, given by ``P(\mathcal{M}) = \int P(\mathcal{M} | \rho) P(\rho) d\rho``.

Bayesian inference does not provide a single estimate of the state, but a full distribution. The mean of the distribution is, in a sense, the best estimate of the state [^1], and the variance gives an idea of the uncertainty of the estimate. Directly calculating the posterior distribution is infeasible, as it requires the computation of the evidence, which is a high-dimensional integral. The alternative is to sample the posterior distribution using Markov Chain Monte Carlo (MCMC) methods. The package provides an implementation of the Metropolis Adjusted Langevin Algorithm (MALA) [^2] [^3] to sample the posterior distribution.

![](assets/random_walk.mp4)

As shown in the video above, MALA generates a random walk in the space of valid density operators (we reject all proposals falling outside this set) whose statistics are given by the desired posterior distribution.


[^1]: [Blume-Kohout, Robin. "Optimal, reliable estimation of quantum states." New Journal of Physics 12.4 (2010): 043034.](https://iopscience.iop.org/article/10.1088/1367-2630/12/4/043034/meta)
[^2]: [Karagulyan, Avetik. Sampling with the Langevin Monte-Carlo. Diss. Institut polytechnique de Paris, 2021.](https://theses.hal.science/tel-03267728/file/103230_KARAGULYAN_2021_archivage.pdf)
[^3]: [Titsias, Michalis. "Optimal Preconditioning and Fisher Adaptive Langevin Sampling." Advances in Neural Information Processing Systems 36 (2024).] (https://arxiv.org/abs/2305.14442)