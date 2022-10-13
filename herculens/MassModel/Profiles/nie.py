# Defines a non-singular isothermal ellipsoid
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
import herculens.Util.util as util
import herculens.Util.param_util as param_util


__all__ = ['NIE', 'NIEMajorAxis']


class NIE(object):
    """
    Non-singular isothermal ellipsoid
    kappa = theta_E/2 [s2IE + r2(1 − e * cos(2*phi)]−1/2
    """
    param_names = ['theta_E', 'e1', 'e2', 's_scale', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 's_scale': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 's_scale': 100, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self):
        self.nie_major_axis = NIEMajorAxis()
        super(NIE, self).__init__()

    def param_conv(self, theta_E, e1, e2, s_scale):
        """
        convert parameters from 2*kappa = bIE [s2IE + r2(1 − e *cos(2*phi)]−1/2 to
        2*kappa=  b *(q2(s2 + x2) + y2􏰉)−1/2
        see expressions after Equation 8 in Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf

        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :return: critical radius b, smoothing scale s, axis ratio q, orientation angle phi_G
        """

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E_conv = self._theta_E_q_convert(theta_E, q)
        b = theta_E_conv * jnp.sqrt((1 + q**2)/2)
        s = s_scale * jnp.sqrt((1 + q**2) / (2*q**2))
        return b, s, q, phi_G

    def function(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        b, s, q, phi_G = self.param_conv(theta_E, e1, e2, s_scale)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self.nie_major_axis.function(x__, y__, b, s, q)
        # rotate back
        return f_

    def derivatives(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        b, s, q, phi_G = self.param_conv(theta_E, e1, e2, s_scale)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = self.nie_major_axis.derivatives(x__, y__, b, s, q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param s_scale: smoothing scale
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_yy, f_xy
        """
        b, s, q, phi_G = self.param_conv(theta_E, e1, e2, s_scale)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__yy, f__xy = self.nie_major_axis.hessian(x__, y__, b, s, q)
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = jnp.cos(2. * phi_G) * gamma1__ - jnp.sin(2. * phi_G) * gamma2__
        gamma2 = jnp.sin(2. * phi_G) * gamma1__ + jnp.cos(2. * phi_G) * gamma2__
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
        theta_E_new = theta_E / (jnp.sqrt((1. + q**2) / (2. * q))) #/ (1+(1-q)/2.)
        return theta_E_new


class NIEMajorAxis(object):
    """
    This class contains the function and the derivatives of the non-singular isothermal ellipse.
    See Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf

    .. math::
        \kappa =  b *(q2(s2 + x2) + y2􏰉)^{−1/2}`

    """

    param_names = ['b', 's', 'q', 'center_x', 'center_y']

    def __init__(self, diff=0.0000000001):  # WARNING: this diff causes large numerical innacuracies in hessian()!
        self._diff = diff
        super(NIEMajorAxis, self).__init__()

    def function(self, x, y, b, s, q):
        psi = self._psi(x, y, q, s)
        alpha_x, alpha_y = self.derivatives(x, y, b, s, q)
        f_ = x * alpha_x + y * alpha_y - b * s * 1. / 2. * jnp.log((psi + s)**2 + (1. - q**2) * x**2)
        return f_

    def derivatives(self, x, y, b, s, q):
        """
        returns df/dx and df/dy of the function
        """
        q = jnp.clip(q, a_max=0.99999999)
        psi = self._psi(x, y, q, s)
        f_x = b / jnp.sqrt(1. - q**2) * jnp.arctan(jnp.sqrt(1. - q**2) * x / (psi + s))
        f_y = b / jnp.sqrt(1. - q**2) * jnp.arctanh(jnp.sqrt(1. - q**2) * y / (psi + q**2 * s))
        return f_x, f_y

    def hessian(self, x, y, b, s, q):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, b, s, q)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, b, s, q)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, b, s, q)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        # f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_yy, f_xy

    @staticmethod
    def kappa(x, y, b, s, q):
        """
        convergence

        :param x: major axis coordinate
        :param y: minor axis coordinate
        :param b: normalization
        :param s: smoothing scale
        :param q: axis ratio
        :return: convergence
        """
        kappa = b/2. * (q**2 * (s**2 + x**2) + y**2)**(-1./2)
        return kappa

    @staticmethod
    def _psi(x, y, q, s):
        """
        expression after equation (8) in Keeton&Kochanek 1998

        :param x: semi-major axis coordinate
        :param y: semi-minor axis coordinate
        :param q: axis ratio minor/major
        :param s: smoothing scale in major axis direction
        :return: phi
        """
        return jnp.sqrt(q**2 * (s**2 + x**2) + y**2)
