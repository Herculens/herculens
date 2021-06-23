# from __future__ import print_function, division, absolute_import, unicode_literals
from jaxtronomy.LensModel.single_plane import SinglePlane
# from jaxtronomy.LensModel.multi_plane import MultiPlane
# from jaxtronomy.Cosmo.lens_cosmo import LensCosmo
# from astropy.cosmology import default_cosmology
# from jaxtronomy.Util import constants as const

__all__ = ['LensModel']


class LensModel(object):
    """An arbitrary list of lens models."""
    def __init__(self, lens_model_list):
        """Create a LensModel object.

        Parameters
        ----------
        lens_model_list : list of str
            Lens model profile names.

        Notes
        -----
        The original LensModel class in lenstronomy has many more inputs and
        supports much more functionality. It has been reduced here to the bare
        minimum in order to test JAX autodiff through a lensing pipeline.

        """
        self.lens_model_list = lens_model_list
        # self.z_lens = z_lens
        # if z_source_convention is None:
        #     z_source_convention = z_source
        # self.z_source = z_source
        # self._z_source_convention = z_source_convention
        self.redshift_list = None  # lens_redshift_list

        # if cosmo is None:
        #     cosmo = default_cosmology.get()
        # self.cosmo = cosmo
        self.multi_plane = False  # multi_plane
        if self.multi_plane:
            if z_source is None:
                raise ValueError('z_source needs to be set for multi-plane lens modelling.')

            self.lens_model = MultiPlane(z_source, lens_model_list,
                                         lens_redshift_list, cosmo=cosmo,
                                         observed_convention_index=observed_convention_index)
        else:
            self.lens_model = SinglePlane(self.lens_model_list,
                                          self.redshift_list)
        # if z_lens is not None and z_source is not None:
        #     self._lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)

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
        return self.lens_model.ray_shooting(x, y, kwargs, k=k)

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
        return self.lens_model.potential(x, y, kwargs, k=k)

    def alpha(self, x, y, kwargs, k=None, diff=None):
        """
        deflection angles

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: None or float. If set, computes the deflection as a finite numerical differential of the lensing
         potential. This differential is only applicable in the single lensing plane where the form of the lensing
         potential is analytically known
        :return: deflection angles in units of arcsec
        """
        if diff is None:
            return self.lens_model.alpha(x, y, kwargs, k=k)
        elif self.multi_plane is False:
            return self._deflection_differential(x, y, kwargs, k=k, diff=diff)
        else:
            raise ValueError('numerical differentiation of lensing potential is not available in the multi-plane '
                             'setting as analytical form of lensing potential is not available.')

    def hessian(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        hessian matrix

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        if diff is None:
            return self.lens_model.hessian(x, y, kwargs, k=k)
        elif diff_method == 'square':
            return self._hessian_differential_square(x, y, kwargs, k=k, diff=diff)
        elif diff_method == 'cross':
            return self._hessian_differential_cross(x, y, kwargs, k=k, diff=diff)
        else:
            raise ValueError('diff_method %s not supported. Chose among "square" or "cross".' % diff_method)

    def kappa(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        lensing convergence k = 1/2 laplacian(phi)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: lensing convergence
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        kappa = (f_xx + f_yy) / 2.
        return kappa

    def curl(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
        """
        curl computation F_yx - F_xy

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: curl at position (x, y)
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        # Note the sign change from lenstronomy
        return f_yx - f_xy

    def gamma(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
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
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: gamma1, gamma2
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        gamma1 = (f_xx - f_yy) / 2.
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, k=None, diff=None, diff_method='square'):
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
        :param diff: float, scale over which the finite numerical differential is computed. If None, then using the
         exact (if available) differentials.
        :param diff_method: string, 'square' or 'cross', indicating whether finite differentials are computed from a
         cross or a square of points around (x, y)
        :return: magnification
        """

        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, k=k, diff=diff, diff_method=diff_method)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx
        return 1. / det_A  # attention, if dividing by zero
