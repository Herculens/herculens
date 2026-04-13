# Defines a uniform mass sheet / external convergence profile
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = "sibirrer, astroskylee"


import jax.numpy as jnp


__all__ = ['Convergence']


class Convergence(object):
    """Uniform mass-sheet / external convergence profile."""

    param_names = ['kappa', 'ra_0', 'dec_0']
    lower_limit_default = {'kappa': -10., 'ra_0': -100., 'dec_0': -100.}
    upper_limit_default = {'kappa': 10., 'ra_0': 100., 'dec_0': 100.}
    fixed_default = {'kappa': False, 'ra_0': True, 'dec_0': True}

    def function(self, x, y, kappa, ra_0=0, dec_0=0):
        """Lensing potential."""
        x_ = x - ra_0
        y_ = y - dec_0
        return 0.5 * kappa * (x_**2 + y_**2)

    def derivatives(self, x, y, kappa, ra_0=0, dec_0=0):
        """Deflection angles (first derivatives)."""
        x_ = x - ra_0
        y_ = y - dec_0
        return kappa * x_, kappa * y_

    def hessian(self, x, y, kappa, ra_0=0, dec_0=0):
        """Return (f_xx, f_yy, f_xy) in the herculens convention."""
        x_ = x - ra_0
        f_xx = kappa + jnp.zeros_like(x_)
        f_yy = kappa + jnp.zeros_like(x_)
        f_xy = jnp.zeros_like(x_)
        return f_xx, f_yy, f_xy
