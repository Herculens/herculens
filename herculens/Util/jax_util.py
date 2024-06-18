# Classes and functions to use with JAX
#
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal', 'duxfrederic'


from copy import deepcopy
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import gammaln


def unjaxify_kwargs(kwargs_params):
    """
    Utility to convert all JAX's device arrays contained in a model kwargs
    to standard floating point or numpy arrays.
    """
    def unjaxify_array(p):
        if p is None or isinstance(p, (float, int)):
            return p  # don't do anything
        if p.size == 1:
            return float(p)
        return np.array(p)
    kwargs_params_new = deepcopy(kwargs_params)
    for model_key, model_kwargs in kwargs_params.items():
        for profile_idx, profile_kwargs in enumerate(model_kwargs):
            for k, p in profile_kwargs.items():
                if isinstance(p, (tuple, list)):
                    # iterate over the list/tuple and unjaxify the items
                    kwargs_params_new[model_key][profile_idx][k] = []  # NOTE: if it was a tuple before, it's not anymore
                    for p_sub in p:
                        kwargs_params_new[model_key][profile_idx][k].append(unjaxify_array(p_sub))
                else:
                    # unjaxify the array
                    kwargs_params_new[model_key][profile_idx][k] = unjaxify_array(p)
    return kwargs_params_new


def R_omega(z, t, q, nmax):
    """Angular dependency of the deflection angle in the EPL lens profile.

    The computation follows Eqs. (22)-(29) in Tessore & Metcalf (2015), where
    z = R * e^(i * phi) is a position vector in the lens plane,
    t = gamma - 1 is the logarithmic slope of the profile, and
    q is the axis ratio.

    This iterative implementation is necessary, since the usual hypergeometric
    function `hyp2f1` provided in `scipy.special` has not yet been implemented
    in an autodiff way in JAX.

    Note that the value returned includes an extra factor R multiplying Eq. (23)
    for omega(phi).

    """
    # Compute constant factors
    f = (1. - q) / (1. + q)
    ei2phi = z / z.conjugate()
    # Set the first term of the series
    omega_i = z  # jnp.array(np.copy(z))  # Avoid overwriting z ?
    partial_sum = omega_i

    @jit
    def body_fun(i, val):
        # Current term in the series is proportional to the previous
        ratio = (2. * i + t - 2.) / (2. * i - t + 2.)
        val[1] = -f * ei2phi * ratio * val[1]
        # Adds the current term to the partial sum
        val[0] += val[1]
        return val

    return lax.fori_loop(1, nmax, body_fun, [partial_sum, omega_i])[0]


def omega_real(x, y, t, q, nmax):
    """Angular dependency of the deflection angle in the EPL lens profile.

    The computation follows Eqs. (31)-(32) in Tessore & Metcalf (2015), where
    x, y are coordinates in the lens plane,
    t = gamma - 1 is the logarithmic slope of the profile, and
    q is the axis ratio.

    This iterative implementation is necessary, since the usual hypergeometric
    function `hyp2f1` provided in `scipy.special` has not yet been implemented
    in an autodiff way in JAX.

    This function is based on the Giga-lens implementation (gigalens.jax.profiles.mass.epl).
    """
    # Compute constant factors
    phi = jnp.arctan2(y, q * x)
    f = (1. - q) / (1. + q)
    Cs, Ss = jnp.cos(phi), jnp.sin(phi)
    Cs2, Ss2 = jnp.cos(2 * phi), jnp.sin(2 * phi)
    def update(n, val):
        prefac = -f * (2. * n - (2. - t)) / (2. * n + (2. - t))
        last_x, last_y, fx, fy = val
        last_x, last_y = prefac * (Cs2 * last_x - Ss2 * last_y), prefac * (
                Ss2 * last_x + Cs2 * last_y
        )
        fx += last_x
        fy += last_y
        return last_x, last_y, fx, fy
    _, _, fx, fy = lax.fori_loop(1, nmax, update, (Cs, Ss, Cs, Ss))
    return fx, fy


class special(object):
    @staticmethod
    @jit
    def gamma(x):
        """Gamma function.

        This function is necessary in many lens mass models, but JAX does not
        currently have an implementation in jax.scipy.special. Values of the
        function are computed simply as the exponential of the logarithm of the
        gamma function (which has been implemented in jax), taking the sign
        of negative inputs properly into account.

        Note that even when just-in-time compiled, this function is much
        slower than its original scipy counterpart.

        Parameters
        ----------
        x : array_like
            Real-valued argument.

        Returns
        -------
        scalar or ndarray
            Values of the Gamma function.
        """
        # Properly account for the sign of negative x values
        sign_condition = (x > 0) | (jnp.ceil(x) % 2 != 0) | (x % 2 == 0)
        sign = 2 * jnp.asarray(sign_condition, dtype=float) - 1
        return sign * jnp.exp(gammaln(x))
