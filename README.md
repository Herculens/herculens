<img src="https://raw.githubusercontent.com/Herculens/herculens/main/images/horizontal.png#gh-light-mode-only" width="600" alt="Herculens logo" />
<img src="https://raw.githubusercontent.com/Herculens/herculens/main/images/horizontal_dark_bg.png#gh-dark-mode-only" width="600" alt="Herculens logo" />

# Herculens: differentiable gravitational lensing

![PyPi python support](https://img.shields.io/badge/Python-3.9%20%7C%203.12-blue)
[![Tests](https://github.com/austinpeel/herculens/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/austinpeel/herculens/actions/workflows/ci_tests.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2207.05763-b31b1b.svg)](https://arxiv.org/abs/2207.05763)
![License](https://img.shields.io/github/license/austinpeel/herculens)
<!-- ![PyPi version](https://img.shields.io/pypi/v/herculens) -->
<!-- [![Coverage Status](https://coveralls.io/repos/github/herculens/herculens/badge.svg?branch=main)](https://coveralls.io/github/aymgal/utax?branch=main) -->

## Analysis of strong lensing imaging data

The primary purpose of `Herculens` is to provide flexible modeling methods to model current and future observations of strong gravitational lenses. Currently, it supports various degrees of model complexity, ranging from standard smooth analytical profiles to pixelated models combined with machine learning approaches.

Currently, `Herculens` supports several of the most widely-used analytical profiles, as well as multi-scale pixelated models regularized with wavelets. Future updates will include the support of point source modeling, new regularization techniques, and more expressive models based on neural networks.

## `JAX`-based automatic differentiation and code compilation 

`Herculens` is based on the powerful framework of **differentiable programming**. The code is entirely based on the automatic differentiation and compilation features of [JAX](https://jax.readthedocs.io/en/latest/#). This simply means that you have access, _analytically_, to all partial derivatives of your model with respect to any of its parameters. This enables faster convergence to the solution, more efficient exploration of the parameter space including the sampling of posterior distributions, and new ways to mitigate degeneracies that affect gravitational lensing.

This highly modular framework offers a way to merge all modeling paradigms explored in the literature, into a single tool:

- **analytical**: model components are described by analytical functions with few parameters and clear physical meaning, but that may be insufficient to fit all observations;
- **pixelated**: regular or irregular grid of pixels are used as individual parameters, which offer higher flexibility, but requires well-motivated regularization strategies;
- **deep learning**: neural networks (among others) are by construction fully differentiable, regardless of being pre-trained or not. It is therefore effortless to plug-in any deep learning-based model component to `Herculens`.

Being based on JAX means that `Herculens` runs effortlessly on CPUs, GPUs and TPUs with no changes to the code.

## Works using `Herculens`

You can find a list of analyses that used `Herculens`, as well as their ADS bibtex entries, on [this page](PUBLICATIONS.md).

## Example notebooks

Several examples to run `Herculens` in many different situations are available in the [`herculens_workspace`](https://github.com/Herculens/herculens_workspace) repository.

## Installation

### Manual installation

The package will be soon available through PyPi directly, but it is as easy to install it manually. It has been texted against Python 3.7, but should work with Python 3.8 or more recent versions.

Good practice is to create a new python environment:
```sh
conda create -n herculens-env python=3.12
conda activate herculens-env
```

Download the package `cd` into the directory. Then install the local `Herculens` package (or use `-e` for a development install) as follows:
```sh
pip install (-e) .
```

The mandatory dependencies will be installed automatically.

### Package requirements

The [`requirements.txt`](requirements.txt) file lists all required and optional package dependencies, along with their specific versions. **IMPORTANT NOTE:** do not only install blindly the required packages; please also have a look at the content of [the file](requirements.txt) itself to learn more about additional optional packages, depending on your goals.

## Attribution

### Citation

If you make use of `Herculens`, please cite [Galan et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A.155G/abstract). If you use the [`CorrelatedField`](https://github.com/Herculens/herculens/blob/main/herculens/GenericModel/correlated_field.py) model, please also cite [Galan et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240218636G/exportcitation). You will find in [this file](CITATION.md) the related bibtex entries.

### Special mention regarding Lenstronomy
Part of the `Herculens` code originates from the open-source lens modeling software package [`lenstronomy`](https://github.com/sibirrer/lenstronomy), described in [Birrer et al. 2021](https://joss.theoj.org/papers/10.21105/joss.03283) (and references therein). In every source file, proper credits are given to the specific developers and contributors to both the original `lenstronomy` (up to version 1.9.3) and `Herculens`.

### Contributors

The list of people that contributed to `Herculens` and credits to original `lenstronomy` contributors, is in [this document](AUTHORS.md).
