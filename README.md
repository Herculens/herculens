<img src="images/horizontal.png" width="600" alt="Herculens logo" />

# Herculens: differentiable gravitational lensing

![License](https://img.shields.io/github/license/austinpeel/herculens)
![PyPi python support](https://img.shields.io/badge/Python-3.7-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2207.05763-b31b1b.svg)](https://arxiv.org/abs/2207.05763)
<!-- ![PyPi version](https://img.shields.io/pypi/v/herculens) -->




## Analysis of strong lensing imaging data

The primary purpose of `herculens` is to provide flexible modeling methods to model current and future observations of strong gravitational lenses. Currently, it supports various degrees of model complexity, ranging from standard smooth analytical profiles to pixelated models combined with machine learning approaches.

Currently, `herculens` supports several of the most widely-used analytical profiles, as well as multi-scale pixelated models regularized with wavelets. Future updates will include the support of point source modeling, new regularization techniques, and more expressive models based on neural networks.





## `JAX`-based automatic differentiation and code compilation 

`herculens` is based on the powerful framework of **differentiable programming**. The code is entirely based on the automatic differentiation and compilation features of [JAX](https://jax.readthedocs.io/en/latest/#). This simply means that you have access, _analytically_, to all partial derivatives of your model with respect to any of its parameters. This enables faster convergence to the solution, more efficient exploration of the parameter space including the sampling of posterior distributions, and new ways to mitigate degeneracies that affect gravitational lensing.

This highly modular framework offers a way to merge all modeling paradigms explored in the literature, into a single tool:

- **analytical**: model components are described by analytical functions with few parameters and clear physical meaning, but that may be insufficient to fit all observations;
- **pixelated**: regular or irregular grid of pixels are used as individual parameters, which offer higher flexibility, but requires well-motivated regularization strategies;
- **deep learning**: neural networks (among others) are by construction fully differentiable, regardless of being pre-trained or not. It is therefore effortless to plug-in any deep learning-based model component to `herculens`.





## Example notebooks

Below you can find a few examples of synthetic problems that can be tackled with `herculens`:

- [**Starting guide**: Lens modeling with gradient descent, Fisher analysis and Hamiltonian sampling](https://github.com/aymgal/herculens_workspace/blob/main/notebooks/herculens__Starting_guide.ipynb)
- [Reconstruction of a complex lensed galaxy with wavelets (and analytical profiles)](https://github.com/aymgal/herculens_workspace/blob/main/notebooks/herculens__Complex_source_with_wavelets.ipynb)
- [Detection of a dark satellite as a localized deviation to the smooth lens potential](https://github.com/aymgal/herculens_workspace/blob/main/notebooks/herculens__Subhalo_detection_with_wavelets.ipynb)

New notebooks will be added soon in order to showcase other features of the package and apply those in different situations.



## Installation

### Manual installation

The package will be soon available through PyPi directly, but it is as easy to install it manually.

Good practice is to create a new python environment:
```sh
conda create -n herculens-env python=3.7
conda activate herculens-env
```

Download the package `cd` into the directory. Then install the local `herculens` package (or use `-e` for a development install) as follows:
```sh
pip install (-e) .
```

The following dependencies will be installed automatically.



### External package requirements

`herculens` has been tested with the versions mentioned in parenthesis. It could still work well with newer versions though.

#### Necessary
- [`jax`](https://github.com/google/jax) (0.3.17)
- `jaxlib` (0.3.14)
- `numpy` (1.21.6)
- `scipy` (1.7.3)

#### Optional
- [`optax`](https://github.com/deepmind/optax) (0.1.3), for advanced gradient descent algorithms 
- [`numpyro`](https://github.com/pyro-ppl/numpyro) (0.10), for HMC sampling
- [`blackjax`](https://github.com/blackjax-devs/blackjax) (0.8.3), for `jax`-based HMC sampling integrated with 
- [`dynesty`](https://github.com/joshspeagle/dynesty) (1.1), for nested sampling
- [`jaxns`](https://github.com/Joshuaalbert/jaxns) (1.1), for `jax`-based nested sampling
- [`emcee`](https://github.com/dfm/emcee) (3.1.2), for Ensemble MCMC sampling
- [`gigalens`](https://github.com/giga-lens/gigalens) (0.1.8), for a `jax` implementation of Shapelets
- [`lenstronomy`](https://github.com/sibirrer/lenstronomy) (1.9.3), for interpolated Shapelets (via `gigalens`) and particle swarm optimization

To run the notebooks, `jupyter` and `matplotlib` are also necessary.



## Attribution

### Citation

If you make use of `herculens`, please cite [Galan et al. 2022](https://arxiv.org/abs/2207.05763). This work presents the global architecture of the tool, and applies it to reconstruct a pixelated lens potential with wavelets.
```
@article{Galan:2022:herculens,
    author        = "Galan, Aymeric and Vernardos, Georgios and Peel, Austin and Courbin, Fr\'ed\'eric and Starck, Jean-Luc",
    title         = "{Using wavelets to capture deviations from smoothness in galaxy-scale strong lenses}",
    eprint        = "2207.05763",
    archivePrefix = "arXiv",
    primaryClass  = "astro-ph.CO",
    year          = "2022"
}
```


### Special mention regarding Lenstronomy
Part of the `herculens` code originates from the open-source lens modeling software package [`lenstronomy`](https://github.com/sibirrer/lenstronomy), described in [Birrer et al. 2021](https://joss.theoj.org/papers/10.21105/joss.03283) (and references therein). In every source file, proper credits are given to the specific developers and contributors to both the original `lenstronomy` (up to version 1.9.3) and `herculens`.


### Contributors

The list of people that contributed to `Herculens` and credits to original `lenstronomy` contributors, is in [this document](AUTHORS.md).
