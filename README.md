# Herculens

## JAX-enabled autodifferentiable strong lens modelling

![Building blocks](images/horizontal.png "Building blocks")

### Recommended Usage
First create an independent python environment.
```sh
conda create -n jax-strong-lensing python=3.7
conda activate jax-strong-lensing
```

Then pip install (develop) the local `herculens` package.
```sh
pip install (-e) .
```

The following dependencies will be installed automatically.

### Requirements (tested version)
- `numpy` (1.20.3)
- `scipy` (1.6.3)
- `jax` (0.2.13)
- `jaxlib` (0.1.67)

##### Optional
- `optax` (0.0.9), for advanced gradient descent algorithms 
- `numpyro` (0.7.1), for HMC sampling
- `emcee` (3.0.2), for MCMC
- `jaxns` (0.0.7), for `jax`-enabled nested sampling
- `dynesty` (1.1), for nested sampling using Hamiltonian slice sampling
- `corner` (2.2.1), for corner plots
- `lenstronomy` (1.6.0), for particle swarm optimization (this might be removed in the future)
- `palettable` (3.3.0), for nicer colormaps for plots

To run the notebooks, `jupyter` is (of course) also necessary, along with `matplotlib` for plotting.

### Notes
The foundation of the `herculens` package implemented here comes from [`lenstronomy`](https://github.com/sibirrer/lenstronomy), a popular strong
gravitational lens modelling software. The original code has been trimmed to the minimum necessary for our purposes. Modifications have been made throughout, primarily to allow JAX autodiff to operate through the workflow. Many stylistic changes have been made as well. The basic module structure, however, remains largely the same.
