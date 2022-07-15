# Defines external shear profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
from herculens.Util import param_util


__all__ = ['Shear', 'ShearGammaPsi']


class Shear(object):
    """External shear in terms of gamma1 and gamma2."""
    param_names = ['gamma1', 'gamma2', 'ra_0', 'dec_0']
    lower_limit_default = {'gamma1': -0.5, 'gamma2': -0.5, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'gamma1': 0.5, 'gamma2': 0.5, 'ra_0': 100, 'dec_0': 100}
    fixed_default = {'gamma1': False, 'gamma2': False, 'ra_0': True, 'dec_0': True}

    def function(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: lensing potential
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 0.5 * (gamma1 * x_ * x_ + 2 * gamma2 * x_ * y_ - gamma1 * y_ * y_)
        return f_

    def derivatives(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: deflection angles
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = gamma1 * x_ + gamma2 * y_
        f_y = gamma2 * x_ - gamma1 * y_
        return f_x, f_y

    def hessian(self, x, y, gamma1, gamma2, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma1: shear component
        :param gamma2: shear component
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return: f_xx, f_yy, f_xy
        """
        gamma1 = gamma1
        gamma2 = gamma2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy


class ShearGammaPsi(object):
    """External shear in terms of magnitude and position angle."""
    param_names = ['gamma_ext', 'psi_ext', 'ra_0', 'dec_0']
    lower_limit_default = {'gamma_ext': 0, 'psi_ext': -np.pi, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'gamma_ext': 1, 'psi_ext': np.pi, 'ra_0': 100, 'dec_0': 100}
    fixed_default = {'gamma_ext': False, 'psi_ext': False, 'ra_0': True, 'dec_0': True}
    
    def __init__(self):
        self._shear_e1e2 = Shear()
        super(ShearGammaPsi, self).__init__()

    @staticmethod
    def function(x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        """

        :param x: x-coordinate (angle)
        :param y: y0-coordinate (angle)
        :param gamma_ext: shear strength
        :param psi_ext: shear angle (radian)
        :param ra_0: x/ra position where shear deflection is 0
        :param dec_0: y/dec position where shear deflection is 0
        :return:
        """
        # change to polar coordinate
        r, phi = param_util.cart2polar(x-ra_0, y-dec_0)
        f_ = 1. / 2 * gamma_ext * r ** 2 * np.cos(2 * (phi - psi_ext))
        return f_

    def derivatives(self, x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        # rotation angle
        gamma1, gamma2 = param_util.shear_polar2cartesian(psi_ext, gamma_ext)
        return self._shear_e1e2.derivatives(x, y, gamma1, gamma2, ra_0, dec_0)

    def hessian(self, x, y, gamma_ext, psi_ext, ra_0=0, dec_0=0):
        gamma1, gamma2 = param_util.shear_polar2cartesian(psi_ext, gamma_ext)
        return self._shear_e1e2.hessian(x, y, gamma1, gamma2, ra_0, dec_0)
