<img src="images/horizontal.png" width="600" alt="Herculens logo" />
<!-- ![Herculens logo](images/horizontal.png "Herculens logo") -->

# Herculens: fully differentiable gravitational lensing

![License](https://img.shields.io/github/license/austinpeel/jax-strong-lensing)
![PyPi version](https://img.shields.io/pypi/v/herculens)
![PyPi python support](https://img.shields.io/pypi/pyversions/herculens)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)



### Analysis of strong lensing imaging data

The primary purpose of `Herculens` is to provide flexible modeling methods to model current and future observations of strong gravitational lenses. Currently, it supports various degrees of model complexity, ranging from standard smooth analytical profiles to pixelated models combined with machine learning approaches.

Currently, `Herculens` supports several of the most widely-used analytical profiles, as well as multi-scale pixelated models regularized with wavelets. Future updates will include the support of point source modeling, new regularization techniques, and more expressive models based on neural networks.

Being a forward modeling code, `Herculens` also allows to simulate realistic observations of gravitational lenses.


### `JAX`-based automatic differentiation and code compilation 

`Herculens` is based on the powerful framework of **differentiable programming**. The code is entirely based on the automatic differentiation and compilation features of [`JAX`](https://github.com/google/jax). This simply means that you have access, _analytically_, to all partial derivatives of your model with respect to any of its parameters. This enables faster convergence to the solution, more efficient exploration of the parameter space including the sampling of posterior distributions, and new ways to mitigate degeneracies that affect gravitational lensing.

This highly modular framework offers a way to merge all modeling paradigms explored in the literature, into a single tool:

- **analytical**: model components are described by analytical functions with few parameters and clear physical meaning, but that can be insufficient to fit all observations;
- **pixelated**: regular or irregular grid of pixels are used as individual parameters, which offer higher flexibility, but requires well-motivated regularization strategies;
- **deep learning**: neural networks (among others) are by construction fully differentiable, regardless of being pre-trained or not. It is therefore effortless to plug-in any deep learning-based model component to `Herculens`.



### Example notebooks

Below you can find a few examples of synthetic problems that can be tackled with `Herculens`:

- [**Starting guide**: Lens modeling with gradient descent, Fisher analysis and Hamiltonian sampling](https://github.com/aymgal/herculens_workspace/edit/main/notebooks/herculens__Starting_guide.ipynb)
- [Reconstruction of a complex lensed galaxy with wavelets (and analytical profiles)](https://github.com/aymgal/herculens_workspace/edit/main/notebooks/herculens__Complex_source_with_wavelets.ipynb)
- [Detection of a dark satellite as a localized deviation to the smooth lens potential](https://github.com/aymgal/herculens_workspace/edit/main/notebooks/herculens__Subhalo_detection_with_wavelets.ipynb)

New notebooks will be added soon in order to showcase other features of the package and apply those in different situations.



### Installation

#### Recommended Usage
First create an independent python environment:
```sh
conda create -n herculens-env python=3.7
conda activate herculens-env
```

Then install the local `herculens` package (or use `-e` for a development install):
```sh
pip install (-e) .
```

The following dependencies will be installed automatically.



#### External package requirements

We tested `Herculens` with the versions mentioned in parenthesis. It does not mean that it breaks for newer versions though.

##### Necessary
- `jax` (0.2.13)
- `jaxlib` (0.1.67)
- `numpy` (1.20.3)
- `scipy` (1.6.3)

##### Optional
- `optax` (0.0.9), for advanced gradient descent algorithms 
- `numpyro` (0.7.1), for HMC sampling
- `blackjax` (0.7.1), for `jax`-based HMC sampling integrated with 
- `dynesty` (1.1), for nested sampling
- `jaxns` (1.1), for `jax`-based nested sampling
- `emcee` (3.0.2), for Ensemble MCMC sampling
- `lenstronomy` (1.6.0), for particle swarm optimization
- `palettable` (3.3.0), for nicer colormaps for plots

To run the notebooks, `jupyter` is (of course) also necessary, along with `matplotlib` for plotting.



### Open-source development

`Herculens` is currently being actively developed, and several science projects will bring many improvements and new features along the way. Moreover, you can expect very soon better documentation and unit testing to increase the robustness of the code.

Anyone that would like to contribute to `Herculens` is highly encouraged to do so. Please do not hesitate to contact us, open issues on github, fork the repository, and suggest improvements!





### A note regarding `lenstronomy`

You may quickly realize that the syntax and naming conventions of `Herculens` share many similarities with the general purpose lensing software package [`lenstronomy`](https://github.com/sibirrer/lenstronomy) ([Birrer et al. 2021](https://joss.theoj.org/papers/10.21105/joss.03283), and references therein). The reason is simple: because it is overall easy to use and open-source, `lenstronomy` was used a as baseline for the early development phases of `Herculens`. Since then, there have been many modifications throughout the code, primarily to add powerful features from `JAX`, then to implement new modeling techniques that were unachievable beforehand. These features could _not_ be implemented in the `lenstronomy` without significantly affecting the spirit of the package. While many stylistic changes have been made, it should be really easy for any `lenstronomy` user to test specific features exclusive to `Herculens`.



### Parallelization and GPU support

One thing that has not been extensively tested yet---although it is automatically supported with `JAX`---is to run `Herculens` in parallel and over multiple GPUs. If you wish to do that, we would be delighted to get touch with you for possible improvements regarding these aspects.
