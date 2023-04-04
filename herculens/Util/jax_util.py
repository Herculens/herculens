# Classes and functions to use with JAX
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal', 'duxfrederic'


from functools import partial
from copy import deepcopy
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.scipy.special import gammaln
from jax.scipy.stats import norm
from jax.lax import conv_general_dilated, conv_dimension_numbers



def unjaxify_kwargs(kwargs_params):
    """
    Utility to convert all JAX's device arrays contained in a model kwargs 
    to standard floating point or numpy arrays.
    """
    kwargs_params_new = deepcopy(kwargs_params)
    for model_key, model_kwargs in kwargs_params.items():
        for profile_idx, profile_kwargs in enumerate(model_kwargs):
            for param_key, param_value in profile_kwargs.items():
                if not isinstance(param_value, (float, int)):
                    if param_value.size == 1:
                        kwargs_params_new[model_key][profile_idx][param_key] = float(param_value)
                    else:
                        kwargs_params_new[model_key][profile_idx][param_key] = np.array(param_value)
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
    # Set the maximum number of iterations
    # nmax = 10
    
    # Compute constant factors
    f = (1. - q) / (1. + q)
    ei2phi = z / z.conjugate()
    # Set the first term of the series
    omega_i = z  # jnp.array(np.copy(z))  # Avoid overwriting z ?
    partial_sum = omega_i

    for i in range(1, nmax):
        # Iteration-dependent factor
        ratio = (2. * i - (2. - t)) / (2. * i + (2 - t))
        # Current Omega term proportional to the previous term
        omega_i = -f * ratio * ei2phi * omega_i
        # Update the partial sum
        partial_sum += omega_i
    return partial_sum


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
