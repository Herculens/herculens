# Defines shapelet light profiles based on the Gaussian-Hermite basis.
#
# Copyright (c) 2021, herculens developers and contributors
#
# The core Gaussian-Hermite shapelet computation (_GaussHermiteShapelets) is
# adapted from the GigaLens project:
#   https://github.com/giga-lens/gigalens/blob/master/src/gigalens/jax/profiles/light/shapelets.py
# Copyright (c) GigaLens and Lenstronomy contributors
# All gigalens / lenstronomy / tensorflow-probability dependencies have been
# removed; the basis is now computed purely with JAX / NumPy.

__author__ = 'aymgal', 'martin-millon'

import numpy as np
import jax
import jax.numpy as jnp


__all__ = ['Shapelets']


class _GaussHermiteShapelets:
    """JAX implementation of the Gaussian-Hermite shapelet basis.

    Computes the 2-D shapelet surface-brightness model

        f(x, y) = sum_{n1, n2} a_{n1,n2}  phi_{n1}(x') phi_{n2}(y')

    where x' = (x - cx) / beta, y' = (y - cy) / beta, and

        phi_n(t) = [2^n sqrt(pi) n!]^{-1/2}  H_n(t)  exp(-t^2 / 2)

    with H_n the physicists' Hermite polynomial.  Basis functions are ordered
    with the same (n1, n2) pairing convention used in GigaLens / lenstronomy.

    Parameters
    ----------
    n_max : int
        Maximum shapelet order.
    interpolate : bool, optional
        If True, phi_n values are looked up from precomputed tables via linear
        interpolation rather than being evaluated via the Hermite recursion on
        every call.  This is faster at call time (and for gradient computation)
        but introduces a small approximation error whose magnitude depends on
        ``grid_limit`` and ``n_grid``.  Default False.
    grid_limit : float, optional
        Half-width of the precomputed table in normalised coordinates
        ``t = (x - cx) / beta``.  Pixels with ``|t| > grid_limit`` are set
        to zero (the Gaussian envelope makes this negligible in practice).
        Only used when ``interpolate=True``.  Default 6.0.
    n_grid : int, optional
        Number of uniformly spaced sample points in ``[-grid_limit, grid_limit]``
        used to build each phi_n table.  Larger values give better accuracy at
        the cost of more memory.  Only used when ``interpolate=True``.
        Default 6000.
    """

    def __init__(self, n_max, interpolate=False, grid_limit=6.0, n_grid=6000):
        self.n_max = n_max
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.interpolate = interpolate

        # Build ordered (n1, n2) index pairs
        N1, N2 = [], []
        n1, n2 = 0, 0
        for _ in range(self.n_layers):
            N1.append(n1)
            N2.append(n2)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        self.N1 = jnp.array(N1, dtype=jnp.int32)
        self.N2 = jnp.array(N2, dtype=jnp.int32)

        # Normalization: prefactor[n] = 1 / sqrt(2^n * sqrt(pi) * n!)
        N = jnp.arange(0, n_max + 1, dtype=jnp.float32)
        self.prefactor = 1.0 / jnp.sqrt(
            2.0 ** N * jnp.sqrt(jnp.pi) * jnp.exp(jax.lax.lgamma(N + 1.0))
        )

        if interpolate:
            self._grid_limit = float(grid_limit)
            self._n_grid = int(n_grid)
            self._phi_tables = self._build_phi_tables()

    # ------------------------------------------------------------------
    # Interpolation mode helpers
    # ------------------------------------------------------------------

    def _build_phi_tables(self):
        """Precompute phi_n(t) on a regular grid for n = 0 .. n_max.

        All computation is done with NumPy at __init__ time (no JAX tracing),
        so this only runs once regardless of how many times ``function`` is called.

        Returns
        -------
        jnp.ndarray, shape (n_max + 1, n_grid)
            ``table[n, k]`` = phi_n evaluated at the k-th grid point.
        """
        t = np.linspace(-self._grid_limit, self._grid_limit, self._n_grid)

        # Hermite polynomial recursion in NumPy (fast, exact, done once)
        H = np.ones((self.n_max + 1, self._n_grid), dtype=np.float64)
        if self.n_max >= 1:
            H[1] = 2.0 * t
        for n in range(2, self.n_max + 1):
            H[n] = 2.0 * (t * H[n - 1] - (n - 1) * H[n - 2])

        # Apply Gaussian envelope and normalization
        fac = np.exp(-t ** 2 / 2.0)
        prefactor_np = np.array(self.prefactor, dtype=np.float64)
        phi = prefactor_np[:, None] * H * fac[None, :]   # (n_max+1, n_grid)

        return jnp.array(phi, dtype=jnp.float32)

    def _interp_phi(self, t):
        """Linearly interpolate phi_n tables at arbitrary positions ``t``.

        Parameters
        ----------
        t : jnp.ndarray, shape (n_pixels,)
            Normalised coordinates (flattened).

        Returns
        -------
        jnp.ndarray, shape (n_max + 1, n_pixels)
            Interpolated phi_n values.  Positions outside
            ``[-grid_limit, grid_limit]`` return 0.
        """
        x_min = -self._grid_limit
        x_max = self._grid_limit
        n_grid = self._n_grid

        # Continuous index into the table
        idx_f = (t - x_min) / (x_max - x_min) * (n_grid - 1)

        # Mask for out-of-range positions (Gaussian makes these ~0 anyway)
        in_range = (idx_f >= 0.0) & (idx_f <= float(n_grid - 1))

        idx_f = jnp.clip(idx_f, 0.0, float(n_grid - 1))
        idx_lo = jnp.floor(idx_f).astype(jnp.int32)
        idx_hi = jnp.minimum(idx_lo + 1, n_grid - 1)

        w_hi = (idx_f - idx_lo.astype(jnp.float32))   # (n_pixels,)
        w_lo = 1.0 - w_hi

        # Gather and interpolate: (n_max+1, n_pixels)
        vals = w_lo * self._phi_tables[:, idx_lo] + w_hi * self._phi_tables[:, idx_hi]

        # Zero out positions that fell outside the table range
        return jnp.where(in_range[None, :], vals, 0.0)

    # ------------------------------------------------------------------
    # Public evaluation
    # ------------------------------------------------------------------

    def function(self, x, y, beta, center_x, center_y, amps):
        """Evaluate the shapelet model at positions (x, y).

        Parameters
        ----------
        x, y : array_like
            Coordinate arrays (arcsec).
        beta : float
            Shapelet scale radius (arcsec).
        center_x, center_y : float
            Centroid position (arcsec).
        amps : array_like, shape (n_layers,)
            Shapelet amplitude coefficients.

        Returns
        -------
        jnp.ndarray
            Surface brightness at each (x, y) position.
        """
        amps = jnp.asarray(amps)
        orig_shape = jnp.asarray(x).shape

        # Normalised coordinates
        xn = (x - center_x) / beta
        yn = (y - center_y) / beta

        if self.interpolate:
            # Look up phi_n from precomputed tables via linear interpolation.
            # Tables include the Gaussian envelope, so no separate fac needed.
            xn_flat = jnp.asarray(xn).ravel()
            yn_flat = jnp.asarray(yn).ravel()

            phi_x = self._interp_phi(xn_flat)   # (n_max+1, n_pixels)
            phi_y = self._interp_phi(yn_flat)   # (n_max+1, n_pixels)

            basis = phi_x[self.N1] * phi_y[self.N2]   # (n_layers, n_pixels)
            result = jnp.einsum('i,ij->j', amps, basis)
            return result.reshape(orig_shape)

        else:
            # Exact Hermite recursion (no approximation, deeper compute graph).
            # Stack x and y so the recursion runs on both at once: (2, *x.shape)
            z = jnp.stack([xn, yn], axis=0)

            H = jnp.ones((self.n_max + 1,) + z.shape)
            H = H.at[0].set(jnp.ones_like(z))
            if self.n_max >= 1:
                H = H.at[1].set(2.0 * z)
            for n in range(2, self.n_max + 1):
                H = H.at[n].set(2.0 * (z * H[n - 1] - (n - 1) * H[n - 2]))

            # Apply normalization: phi has shape (n_max+1, 2, *x.shape)
            phi = jnp.einsum('i,i...->i...', self.prefactor, H)

            phi_x = phi[:, 0, ...]   # (n_max+1, *x.shape)
            phi_y = phi[:, 1, ...]

            fac = jnp.exp(-(xn ** 2 + yn ** 2) / 2.0)
            basis = phi_x[self.N1] * phi_y[self.N2]   # (n_layers, *x.shape)
            return fac * jnp.einsum('i,i...->...', amps, basis)


class Shapelets(object):
    """Surface brightness modelled as a sum of Gaussian-Hermite shapelets.

    Parameters
    ----------
    n_max : int
        Maximum shapelet order.  The total number of basis functions is
        ``(n_max + 1) * (n_max + 2) / 2``.
    function_type : str, optional
        Basis function family.  Currently only ``'gaussian'`` is supported.
    interpolate : bool, optional
        If True, use precomputed phi_n tables with linear interpolation at
        call time instead of running the Hermite recursion.  Faster per call
        and cheaper to differentiate, but introduces a small approximation
        error.  Default True.
    grid_limit : float, optional
        Half-width of the interpolation table in normalised coordinates.
        Ignored when ``interpolate=False``.  Default 6.0.
    n_grid : int, optional
        Number of sample points in the interpolation table.
        Ignored when ``interpolate=False``.  Default 6000.
    """

    param_names = ['beta', 'center_x', 'center_y', 'amps']
    lower_limit_default = {'beta': 0.,   'center_x': -100., 'center_y': -100., 'amps': -1e10}
    upper_limit_default = {'beta': 1e5,  'center_x':  100., 'center_y':  100., 'amps':  1e10}
    fixed_default = {key: False for key in param_names}

    def __init__(self, n_max, function_type='gaussian',
                 interpolate=True, grid_limit=6.0, n_grid=6000):
        if function_type == 'gaussian':
            self._n_max = n_max
            self._backend = _GaussHermiteShapelets(
                n_max,
                interpolate=interpolate,
                grid_limit=grid_limit,
                n_grid=n_grid,
            )
        else:
            raise NotImplementedError(f"Basis function type '{function_type}' is not supported")
        self._func_type = function_type

    @property
    def maximum_order(self):
        return self._n_max

    @property
    def num_amplitudes(self):
        if self._func_type == 'gaussian':
            return int((self._n_max + 1) * (self._n_max + 2) / 2)

    def function(self, x, y, beta, center_x, center_y, amps):
        """Evaluate the shapelet surface brightness at positions (x, y).

        Parameters
        ----------
        x, y : array_like
            Coordinate arrays (arcsec).
        beta : float
            Shapelet scale radius (arcsec).
        center_x, center_y : float
            Centroid position (arcsec).
        amps : array_like, shape (n_amplitudes,)
            Shapelet amplitude coefficients.

        Returns
        -------
        jnp.ndarray
            Surface brightness at each (x, y) position.
        """
        return self._backend.function(x, y, beta, center_x, center_y, amps)
