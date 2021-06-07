# jax-strong-lensing

Recommended usage
```sh
conda create -n jax-strong-lensing python=3.7
conda activate jax-strong-lensing
```

Then pip install the following Python packages.

Requirements (tested with version)
- numpy (1.20.3)
- scipy (1.6.3)
- jax (0.2.13)
- jaxlib (0.1.67)
- matplotlib (3.4.2)


The foundation of the `jaxtronomy` package implemented here comes from [`lenstronomy`](https://github.com/sibirrer/lenstronomy), a popular strong
gravitational lens modelling software. The original code has been trimmed to the minimum necessary for our purposes. Modifications have been made throughout primarily to allow JAX autodiff to operate through the workflow. Many stylistic changes have been made as well. The basic module structure, however, remains largely the same.
