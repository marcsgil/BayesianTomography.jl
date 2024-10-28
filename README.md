# BayesianTomography.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://marcsgil.github.io/BayesianTomography.jl/dev/)
[![CI](https://github.com/marcsgil/BayesianTomography.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/marcsgil/BayesianTomography.jl/actions/workflows/CI.yml)
[![DOI](https://zenodo.org/badge/740175693.svg)](https://zenodo.org/doi/10.5281/zenodo.10936092)

WARNING: This package has been deprecated in favor of [QuantumMeasurements.jl](https://github.com/marcsgil/QuantumMeasurements.jl). Please use the new package for future developments.

This package provides tools to perform quantum state tomography. It is designed to be flexible and easy to use, with a focus on providing a simple interface for common tasks. As its name suggests, its flagship feature is the ability to perform Bayesian quantum state tomography, but it also provides an interface for the simple linear inversion method.

This package was developed as part of the following publication: [Quantum tomography of structured light patterns from simple intensity measurements](https://arxiv.org/abs/2404.05616). Nonetheless, we tried to make it as general as possible, in the hope that it can be useful for a wide range of applications.

## Installation

The package is registered in the Julia General registry and can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run `add BayesianTomography`.

## Usage
For a quick start, check the [documentation](https://marcsgil.github.io/BayesianTomography.jl/dev/usage/).