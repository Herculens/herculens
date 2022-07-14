# Defines Sersic profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'jiwoncpark', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from functools import partial

from herculens.MassModel.Profiles.sersic_utils import SersicUtil
import herculens.Util.param_util as param_util


__all__ = ['Sersic', 'SersicElliptic']


class Sersic(SersicUtil):
    """
    this class contains functions to evaluate an spherical Sersic function

    .. math::

        I(R) = I_0 \exp \left[ -b_n (R/R_{\rm Sersic})^{\frac{1}{n}}\right]

    with :math:`I_0 = amp`
    and
    with :math:`b_{n}\approx 1.999\,n-0.327`

    """

    param_names = ['amp', 'R_sersic', 'n_sersic', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def function(self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """
        R = self.get_distance_from_center(x, y, phi_G=0.0, q=1.0, center_x=center_x, center_y=center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result

    def derivatives(self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: partial derivatives of Sersic profile value at (x, y) with respect to x and y
        """
        def _function(p):
            return self.function(p[0], p[1], 
                                 amp, R_sersic, n_sersic,
                                 center_x=center_x, center_y=center_y,
                                 max_R_frac=max_R_frac)
        
        grad_function = grad(_function)

        @jit
        def _grad_function(x, y):
            return grad_function([x, y])[0], grad_function([x, y])[1]
        
        f_x, f_y = jnp.vectorize(_grad_function)(x, y)
        return f_x, f_y

    def derivatives_explicit(self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: partial derivatives of Sersic profile value at (x, y) with respect to x and y
        """
        x_ = np.array(x) - center_x
        y_ = np.array(y) - center_y
        r = np.sqrt(x_**2 + y_**2)
        #if isinstance(r, int) or isinstance(r, float):
        #    r = max(self._s, r)
        #else:
        #    r[r < self._s] = self._s
        alpha = -self.alpha_abs(x, y, n_sersic, R_sersic, amp, center_x, center_y)
        f_x = alpha * x_ / r
        f_y = alpha * y_ / r
        return f_x, f_y


class SersicElliptic(SersicUtil):
    """
    this class contains functions to evaluate an elliptical Sersic function
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5,'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8, 'e1': 0.5, 'e2': 0.5,'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def function(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """

        R_sersic = jnp.maximum(0, R_sersic)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        R = self.get_distance_from_center(x, y, phi_G, q, center_x, center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result

    def derivatives(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: partial derivatives of Sersic profile value at (x, y) with respect to x and y
        """
        def _function(p):
            return self.function(p[0], p[1], 
                                 amp, R_sersic, n_sersic, e1, e2,
                                 center_x=center_x, center_y=center_y,
                                 max_R_frac=max_R_frac)
        
        grad_function = grad(_function)

        @jit
        def _grad_function(x, y):
            return grad_function([x, y])[0], grad_function([x, y])[1]
        
        f_x, f_y = jnp.vectorize(_grad_function)(x, y)
        return f_x, f_y
