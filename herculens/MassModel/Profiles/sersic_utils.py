# Utility methods for Sersic profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'jiwoncpark', 'austinpeel', 'aymgal'


import scipy.special as special
import numpy as np
import jax.numpy as jnp
import scipy
from herculens.Util import param_util


__all__ = ['SersicUtil']


class SersicUtil(object):

    _s = 0.00001

    def __init__(self, smoothing=_s):
        self._smoothing = smoothing

    def k_bn(self, n, Re):
        """
        returns normalisation of the sersic profile such that Re is the half light radius given n_sersic slope
        """
        bn = self.b_n(n)
        k = bn*Re**(-1./n)
        return k, bn

    def k_Re(self, n, k):
        """

        """
        bn = self.b_n(n)
        Re = (bn/k)**n
        return Re

    @staticmethod
    def b_n(n):
        """
        b(n) computation. This is the approximation of the exact solution to the relation, 2*incomplete_gamma_function(2n; b_n) = Gamma_function(2*n).
        :param n: the sersic index
        :return:
        """
        bn = 1.9992*n - 0.3271
        return bn

    def get_distance_from_center(self, x, y, phi_G, q, center_x, center_y):
        """
        Get the distance from the center of Sersic, accounting for orientation and axis ratio
        :param x:
        :param y:
        :param phi_G: orientation angle in rad
        :param q: axis ratio
        :param center_x: center x of sersic
        :param center_y: center y of sersic
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = jnp.cos(phi_G)
        sin_phi = jnp.sin(phi_G)
        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        R = jnp.sqrt(xt1*xt1+xt2*xt2difq2)
        return R

    def _x_reduced(self, x, y, n_sersic, r_eff, center_x, center_y):
        """
        coordinate transform to normalized radius
        :param x:
        :param y:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        x_reduced = (r/r_eff)**(1./n_sersic)
        return x_reduced

    def _alpha_eff(self, r_eff, n_sersic, k_eff):
        """
        deflection angle at r_eff
        :param r_eff:
        :param n_sersic:
        :param k_eff:
        :return:
        """
        b = self.b_n(n_sersic)
        alpha_eff = n_sersic * r_eff * k_eff * b**(-2*n_sersic) * jnp.exp(b) * special.gamma(2*n_sersic)
        return -alpha_eff

    def alpha_abs(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param n_sersic:
        :param r_eff:
        :param k_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        n = n_sersic
        x_red = self._x_reduced(x, y, n_sersic, r_eff, center_x, center_y)
        b = self.b_n(n_sersic)
        a_eff = self._alpha_eff(r_eff, n_sersic, k_eff)
        #alpha = 2. * a_eff * x_red ** (-n) * (special.gammainc(2 * n, b * x_red))
        alpha = 2. * a_eff * x_red ** (-n) * (1. - special.gammainc(2 * n, b * x_red)/special.gamma(2 * n))
        return alpha

    def d_alpha_dr(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param n_sersic:
        :param r_eff:
        :param k_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        _dr = 0.00001
        x_ = x - center_x
        y_ = y - center_y
        r = jnp.sqrt(x_**2 + y_**2)
        alpha = self.alpha_abs(r, 0, n_sersic, r_eff, k_eff)
        alpha_dr = self.alpha_abs(r+_dr, 0, n_sersic, r_eff, k_eff)
        d_alpha_dr = (alpha_dr - alpha)/_dr
        return d_alpha_dr

    def density(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        de-projection of the Sersic profile based on
        Prugniel & Simien (1997)
        :return:
        """
        raise ValueError("not implemented! Use a Multi-Gaussian-component decomposition.")

    def _total_flux(self, r_eff, I_eff, n_sersic):
        """
        computes total flux of a round Sersic profile

        :param r_eff: projected half light radius
        :param I_eff: surface brightness at r_eff (in same units as r_eff)
        :param n_sersic: Sersic index
        :return: integrated flux to infinity
        """
        bn = self.b_n(n_sersic)
        return I_eff * r_eff**2 * 2 * jnp.pi * n_sersic * jnp.exp(bn) / bn**(2*n_sersic) * scipy.special.gamma(2*n_sersic)

    def total_flux(self, amp, R_sersic, n_sersic, e1=0, e2=0, Re=None, gamma=None, center_x=None, center_y=None,
                   alpha=None):
        """
        computes analytical integral to compute total flux of the Sersic profile

        :param amp: amplitude parameter in Sersic function (surface brightness at R_sersic
        :param R_sersic: half-light radius in semi-major axis
        :param Re: Cored Sersic function parameter (optional)
        :param n_sersic: Sersic index
        :param gamma: Cored Sersic function parameter (optional)
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: profile center (ignored)
        :param center_y: profile center (ignored)
        :param alpha: Cored Sersic function parameter (optional)
        :return: Analytic integral of the total flux of the Sersic profile
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        # compute product average half-light radius
        r_eff = R_sersic * jnp.sqrt(q)
        return self._total_flux(r_eff=r_eff, I_eff=amp, n_sersic=n_sersic)

    def _R_stable(self, R):
        """
        Floor R_ at self._smoothing for numerical stability
        :param R: radius
        :return: smoothed and stabilized radius
        """
        return jnp.maximum(self._smoothing, R)

    def _r_sersic(self, R, R_sersic, n_sersic, max_R_frac=100.0, alpha=1.0, R_break=0.0):
        """

        :param R: radius (array or float)
        :param R_sersic: Sersic radius (half-light radius)
        :param n_sersic: Sersic index (float)
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: kernel of the Sersic surface brightness at R
        """
        # Must avoid item assignment on JAX arrays
        R_ = self._R_stable(R)
        R_sersic_ = self._R_stable(R_sersic)
        bn = self.b_n(n_sersic)
        R_frac = R_ / R_sersic_
        good_inds = (jnp.asarray(R_frac) <= max_R_frac).astype(int)
        result = good_inds * jnp.exp(-bn * (R_frac**(1. / n_sersic) - 1.))
        return jnp.nan_to_num(result)
