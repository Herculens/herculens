# Defines en elliptical power law profile
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'dangilman', 'ntessore', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
from herculens.Util import util, param_util, jax_util


__all__ = ['EPL', 'EPLMajorAxis']


class EPL(object):
    """
    Elliptical Power Law
    kappa = (2-t)/2*(b/r)^t
    where t = gamma - 1
    """
    param_names = ['gamma', 'theta_E', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'gamma': 1, 'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'gamma': 3, 'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self):
        self.epl_major_axis = EPLMajorAxis()
        super(EPL, self).__init__()

    def param_conv(self, theta_E, e1, e2, gamma):
        """
        convert parameters from R = r sqrt(1 − e*cos(2*phi)) to
        R = sqrt(q^2 x^2 + y^2)

        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param t: power law slope
        :return: critical radius b, slope t, axis ratio q, orientation angle phi_G
        """

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E_conv = self._theta_E_q_convert(theta_E, q)
        b = theta_E_conv * jnp.sqrt((1. + q**2) / 2.)
        t = gamma - 1.
        return b, t, q, phi_G

    def function(self, x, y, theta_E, e1, e2, gamma, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param t: power law slope
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        b, t, q, phi_G = self.param_conv(theta_E, e1, e2, gamma)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self.epl_major_axis.function(x__, y__, b, t, q)
        # rotate back
        return f_

    def derivatives(self, x, y, theta_E, e1, e2, gamma, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param t: power law slope
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        b, t, q, phi_G = self.param_conv(theta_E, e1, e2, gamma)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = self.epl_major_axis.derivatives(x__, y__, b, t, q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, e1, e2, gamma, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param t: power law slope
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_yy, f_xy
        """

        b, t, q, phi_G = self.param_conv(theta_E, e1, e2, gamma)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__yy, f__xy = self.epl_major_axis.hessian(x__, y__, b, t, q)
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = jnp.cos(2 * phi_G) * gamma1__ - jnp.sin(2 * phi_G) * gamma2__
        gamma2 = +jnp.sin(2 * phi_G) * gamma1__ + jnp.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def _theta_E_q_convert(self, theta_E, q):
        """
        converts a spherical averaged Einstein radius to an elliptical (major axis) Einstein radius.
        This then follows the convention of the SPEMD profile in lenstronomy.
        (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]

        :param theta_E: Einstein radius in lenstronomy conventions
        :param q: axis ratio minor/major
        :return: theta_E in convention of kappa=  b *(q2(s2 + x2) + y2􏰉)−1/2
        """
        theta_E_new = theta_E / (jnp.sqrt((1. + q**2) / (2. * q)))
        return theta_E_new


class EPLMajorAxis(object):
    """
    This class contains the function and the derivatives of the
    elliptical power law.

    kappa = (2-t)/2 * [b/sqrt(q^2 x^2 + y^2)]^t
    where t = gamma - 1 (from EPL class)
    Tessore & Metcalf (2015), https://arxiv.org/abs/1507.01819
    """
    param_names = ['b', 't', 'q', 'center_x', 'center_y']

    def __init__(self):

        super(EPLMajorAxis, self).__init__()

    def function(self, x, y, b, t, q):
        """
        returns the lensing potential
        """
        # deflection from method
        alpha_x, alpha_y = self.derivatives(x, y, b, t, q)

        # deflection potential, eq. (15)
        psi = (x * alpha_x + y * alpha_y) / (2. - t)

        return psi

    def derivatives(self, x, y, b, t, q):
        """
        returns the deflection
        """
        # elliptical radius, eq. (5)
        z = q * x + y * 1j
        R = jnp.abs(z)

        # deflection, eq. (22)
        alpha = 2. / (1. + q) * (b / R)**t * jax_util.R_omega(z, t, q, nmax=20)

        # return real and imaginary part
        alpha_real = jnp.nan_to_num(alpha.real, posinf=10**10, neginf=-10**10)
        alpha_imag = jnp.nan_to_num(alpha.imag, posinf=10**10, neginf=-10**10)

        return alpha_real, alpha_imag

    def hessian(self, x, y, b, t, q):
        """
        returns the Hessian matrix of the lensing potential
        """
        R = jnp.hypot(q * x, y)
        r = jnp.hypot(x, y)

        cos, sin = x / r, y / r
        cos2, sin2 = cos * cos * 2 - 1., sin * cos * 2

        # convergence, eq. (2)
        kappa = (2. - t) / 2. * (b / R)**t
        kappa = jnp.nan_to_num(kappa, posinf=10**10, neginf=-10**10)

        # deflection via method
        alpha_x, alpha_y = self.derivatives(x, y, b, t, q)

        # shear, eq. (17), corrected version from arXiv/corrigendum
        gamma_1 = (1. - t) * (alpha_x * cos - alpha_y * sin) / r - kappa * cos2
        gamma_2 = (1. - t) * (alpha_y * cos + alpha_x * sin) / r - kappa * sin2
        gamma_1 = jnp.nan_to_num(gamma_1, posinf=10**10, neginf=-10**10)
        gamma_2 = jnp.nan_to_num(gamma_2, posinf=10**10, neginf=-10**10)

        # second derivatives from convergence and shear
        f_xx = kappa + gamma_1
        f_yy = kappa - gamma_1
        f_xy = gamma_2

        return f_xx, f_yy, f_xy
