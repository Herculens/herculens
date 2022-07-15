# High-level interface to a mass model
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

from herculens.MassModel.mass_model_base import MassProfileBase


__all__ = ['MassModel']


class MassModel(MassProfileBase):
    """An arbitrary list of lens models."""
    def __init__(self, mass_model_list, kwargs_pixelated={}):
        """Create a MassModel object.

        Parameters
        ----------
        mass_model_list : list of str
            Lens model profile names.
        kwargs_pixelated : dictionary for settings related to PIXELATED profiles.

        Notes
        -----
        The original MassModel class in lenstronomy has many more inputs and
        supports much more functionality. It has been reduced here to the bare
        minimum in order to test JAX autodiff through a lensing pipeline.

        """
        self.profile_type_list = mass_model_list
        super().__init__(self.profile_type_list, kwargs_pixelated=kwargs_pixelated)

    def ray_shooting(self, x, y, kwargs, k=None):
        """
        maps image to source position (inverse deflection)
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: source plane positions corresponding to (x, y) in the image plane
        """
        dx, dy = self.alpha(x, y, kwargs, k=k)
        return x - dx, y - dy

    def fermat_potential(self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None):
        """
        fermat potential (negative sign means earlier arrival time)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """

        potential = self.potential(x_image, y_image, kwargs_lens, k=k)
        if x_source is None or y_source is None:
            x_source, y_source = self.ray_shooting(x_image, y_image, kwargs_lens, k=k)
        geometry = ((x_image - x_source)**2 + (y_image - y_source)**2) / 2.
        return geometry - potential

    def potential(self, x, y, kwargs, k=None):
        """
        lensing potential
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """
        # x = np.array(x, dtype=float)
        # y = np.array(y, dtype=float)
        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        potential = 0.
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                potential += func.function(x, y, **kwargs[i])
        return potential

    def alpha(self, x, y, kwargs, k=None):

        """
        deflection angles
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: deflection angles in units of arcsec
        """
        # x = np.array(x, dtype=float)
        # y = np.array(y, dtype=float)
        if isinstance(k, int):
            return self.func_list[k].derivatives(x, y, **kwargs[k])
        bool_list = self._bool_list(k)
        f_x, f_y = 0., 0.
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_x_i, f_y_i = func.derivatives(x, y, **kwargs[i])
                f_x += f_x_i
                f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, kwargs, k=None):
        """
        hessian matrix
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        # x = np.array(x, dtype=float)
        # y = np.array(y, dtype=float)
        if isinstance(k, int):
            f_xx, f_yy, f_xy = self.func_list[k].hessian(x, y, **kwargs[k])
            return f_xx, f_xy, f_xy, f_yy

        bool_list = self._bool_list(k)
        f_xx, f_yy, f_xy = 0., 0., 0.
        for i, func in enumerate(self.func_list):
            if bool_list[i] is True:
                f_xx_i, f_yy_i, f_xy_i = func.hessian(x, y, **kwargs[i])
                f_xx += f_xx_i
                f_yy += f_yy_i
                f_xy += f_xy_i
        f_yx = f_xy
        return f_xx, f_xy, f_yx, f_yy

    def kappa(self, x, y, kwargs, k=None):
        """
        lensing convergence k = 1/2 laplacian(phi)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing convergence
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        kappa = (f_xx + f_yy) / 2.
        return kappa

    def curl(self, x, y, kwargs, k=None):
        """
        curl computation F_yx - F_xy

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: curl at position (x, y)
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        # Note the sign change from lenstronomy
        return f_yx - f_xy

    def gamma(self, x, y, kwargs, k=None):
        """
        shear computation
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: gamma1, gamma2
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        gamma1 = (f_xx - f_yy) / 2.
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, k=None):
        """
        magnification
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
        return 1. / det_A  # attention, if dividing by zero
