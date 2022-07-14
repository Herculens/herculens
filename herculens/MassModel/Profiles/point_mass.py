# Defines a point mass
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
import jax.numpy as jnp


__all__ = ['PointMass']


class PointMass(object):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['theta_E', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        self.r_min = 10e-25
        super(PointMass, self).__init__()

    def function(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: lensing potential
        """
        x_ = x - center_x
        y_ = y - center_y
        r_ = jnp.sqrt(x_**2 + y_**2)
        r  = jnp.clip(r_, a_min=self.r_min)
        phi = theta_E**2*np.log(r)
        return phi

    def derivatives(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: deflection angle (in radian)
        """
        x_ = x - center_x
        y_ = y - center_y
        r_ = jnp.sqrt(x_**2 + y_**2)
        r  = jnp.clip(r_, a_min=self.r_min)
        alpha = theta_E**2/r
        return alpha*x_/r, alpha*y_/r

    def hessian(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: hessian matrix (in radian)
        """
        x_ = x - center_x
        y_ = y - center_y
        C = theta_E**2
        r2_ = x_**2 + y_**2
        r2  = jnp.clip(r2_, a_min=self.r_min**2)
        f_xx = C * (y_**2-x_**2)/r2**2
        f_yy = C * (x_**2-y_**2)/r2**2
        f_xy = -C * 2*x_*y_/r2**2
        return f_xx, f_yy, f_xy
