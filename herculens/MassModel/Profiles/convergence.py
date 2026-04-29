# Defines a mass sheet (external convergence) profile
#
# Copyright (c) 2024, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel.Profiles module from lenstronomy

__author__ = 'sibirrer', 'martin-millon', 'astroskylee'


__all__ = ['Convergence']


class Convergence(object):
    """A single mass sheet (external convergence)."""
    param_names = ['kappa', 'ra_0', 'dec_0']
    lower_limit_default = {'kappa': -10, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'kappa': 10, 'ra_0': 100, 'dec_0': 100}
    fixed_default = {'kappa': False, 'ra_0': True, 'dec_0': True}

    def function(self, x, y, kappa, ra_0=0, dec_0=0):
        """Lensing potential.

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param kappa: (external) convergence
        :param ra_0: x/ra position where the potential is zero
        :param dec_0: y/dec position where the potential is zero
        :return: lensing potential
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 0.5 * kappa * (x_ * x_ + y_ * y_)
        return f_

    def derivatives(self, x, y, kappa, ra_0=0, dec_0=0):
        """Deflection angles.

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param kappa: (external) convergence
        :param ra_0: x/ra position where deflection is zero
        :param dec_0: y/dec position where deflection is zero
        :return: deflection angles
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = kappa * x_
        f_y = kappa * y_
        return f_x, f_y

    def hessian(self, x, y, kappa, ra_0=0, dec_0=0):
        """Hessian matrix of the lensing potential.

        :param x: x-coordinate (angle)
        :param y: y-coordinate (angle)
        :param kappa: (external) convergence
        :param ra_0: x/ra position where deflection is zero
        :param dec_0: y/dec position where deflection is zero
        :return: f_xx, f_yy, f_xy
        """
        f_xx = kappa
        f_yy = kappa
        f_xy = 0.
        return f_xx, f_yy, f_xy
