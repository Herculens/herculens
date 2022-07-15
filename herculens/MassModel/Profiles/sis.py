# Defines a singular isothermal sphere
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
from herculens.MassModel.Profiles.sie import SIE


__all__ = ['SIS']


class SIS(object):
    """Singular isothermal sphere mass profile."""
    param_names = ['theta_E', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}
    
    def __init__(self):
        self.profile = SIE()
        self._e = 1e-4
        super(SIS, self).__init__()

    def function(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.profile.function(x, y, theta_E, self._e, self._e, center_x, center_y)

    def derivatives(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.profile.derivatives(x, y, theta_E, self._e, self._e, center_x, center_y)

    def hessian(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.profile.hessian(x, y, theta_E, self._e, self._e, center_x, center_y)

    @staticmethod
    def theta2rho(theta_E):
        """
        converts projected density parameter (in units of deflection) into 3d density parameter
        :param theta_E:
        :return:
        """
        self.profile.theta2rho(theta_E)

    @staticmethod
    def mass_3d(r, rho0):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius in angular units
        :param rho0: density at angle=1
        :return: mass in angular units
        """
        self.profile.mass_3d(r, rho0)

    def mass_3d_lens(self, r, theta_E):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units

        :param r: radius in angular units
        :param theta_E: Einstein radius
        :return: mass in angular units
        """
        return self.profile.mass_3d_lens(r, theta_E)

    def mass_2d(self, r, rho0):
        """
        mass enclosed projected 2d sphere of radius r
        :param r:
        :param rho0:
        :param a:
        :param s:
        :return:
        """
        return self.profile.mass_2d(r, rho0)

    def mass_2d_lens(self, r, theta_E):
        """

        :param r:
        :param theta_E:
        :return:
        """
        return self.profile.mass_2d_lens(r, theta_E)

    def grav_pot(self, x, y, rho0, center_x=0, center_y=0):
        """
        gravitational potential (modulo 4 pi G and rho0 in appropriate units)
        :param x:
        :param y:
        :param rho0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.profile.grav_pot(x, y, rho0, center_x=center_x, center_y=center_y)

    def density_lens(self, r, theta_E):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius in angles
        :param theta_E: Einstein radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: density
        """
        return self.profile.density_lens(r, theta_E)

    @staticmethod
    def density(r, rho0):
        """
        computes the density
        :param r: radius in angles
        :param rho0: density at angle=1
        :return: density at r
        """
        return self.profile.density(rho0)

    @staticmethod
    def density_2d(x, y, rho0, center_x=0, center_y=0):
        """
        projected density
        :param x:
        :param y:
        :param rho0:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.profile.density_2d(x, y, rho0, center_x=center_x, center_y=center_y)
