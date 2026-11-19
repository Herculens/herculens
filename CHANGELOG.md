# Changelog

## 0.4.0 (2026-11-19)

### Compatibility

* Support for JAX 0.10.x, 0.11.x and 0.12.x (tested with JAX >= 0.7 and < 0.13): `jnp.clip` calls now use the `min`/`max` keywords instead of the removed `a_min`/`a_max` ([#95](https://github.com/Herculens/herculens/issues/95)).
* Require Python >= 3.11 (as required by JAX >= 0.7).
* Continuous integration now tests the oldest (Python 3.11, JAX 0.7.0) and newest (Python 3.12, latest allowed JAX) supported versions, plus a weekly check against the newest JAX release.
* Require `utax>=0.0.3` (which has the same `jnp.clip` fix).
* Optional `numpyro` dependency now requires `numpyro>=0.21.0` (needed for JAX >= 0.10).
* Remove the `objax` dependency (it was used only for image down-sampling in `Util.image_util.re_size`).
* `matplotlib` is now a required dependency (it was optional, but `import herculens` needs it), and `jax` is restricted to `<0.13`.
* `jax_cosmo` is no longer a required dependency; it is only needed to pass a `jax_cosmo` cosmology to `MPMassModel` (astropy cosmologies work without it).

### New features

* Time-delay computation for multi-plane lensing (`MPMassModel`).
* Multi-plane point source option, and multi-plane plotting.
* New `CONVERGENCE` mass profile.
* Point sources are now rendered on the supersampled grid.

### Fixes

* Adaptive pixelated grid with bilinear interpolation.
* `MPMassModel` failing when `jax_cosmo` is not installed.
* Small fixes in the plotting routines.
* Replace deprecated `scipy.ndimage.morphology` imports.

## 0.1.0 (2024-07-30)

* First release on PyPI.
