# Defines a pixelated profile in the lens potential
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

from herculens.Util.jax_util import BicubicInterpolator


__all__ = ['PixelatedPotential', 'PixelatedPotentialDirac']


class PixelatedPotential(object):
    param_names = ['pixels']
    lower_limit_default = {'pixels': -1e10}
    upper_limit_default = {'pixels': 1e10}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        """Lensing potential on a fixed coordinate grid."""
        super(PixelatedPotential, self).__init__()
        self.x_coords, self.y_coords = None, None

    def function(self, x, y, pixels):
        """Interpolated evaluation of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        # Due to matching scipy's interpolation, we need to switch x and y
        # coordinates as well as transpose
        interp = BicubicInterpolator(self.y_coords, self.x_coords, pixels)
        return interp(y, x)

    def derivatives(self, x, y, pixels):
        """Spatial first derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        interp = BicubicInterpolator(self.y_coords, self.x_coords, pixels)
        return interp(y, x, dy=1), interp(y, x, dx=1)

    def hessian(self, x, y, pixels):
        """Spatial second derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        interp = BicubicInterpolator(self.y_coords, self.x_coords, pixels)
        # TODO Why doesn't this follow the pattern of the first derivatives ?
        psi_xx = interp(y, x, dx=2)
        psi_yy = interp(y, x, dy=2)
        psi_xy = interp(y, x, dx=1, dy=1)
        return psi_xx, psi_yy, psi_xy

    def set_data_pixel_grid(self, pixel_axes):
        self.x_coords, self.y_coords = pixel_axes



class PixelatedPotentialDirac(object):
    param_names = ['psi', 'center_x', 'center_y']
    lower_limit_default = {'psi': -1e10, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'psi': 1e10, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        """Dirac impulse in potential on a fixed coordinate grid."""
        super(PixelatedPotentialDirac, self).__init__()
        self.pp = PixelatedPotential()

    def function(self, x, y, psi, center_x, center_y):
        """Interpolated evaluation of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential.
        psi : float
            Value of the pixelated lensing potential at x, y
        center_x : float
            center in x-coordinate
        center_y : float
            center in y-coordinate

        """
        return jnp.where((x >= center_x - self.hss_x) & 
                         (x <= center_x + self.hss_x) & 
                         (y >= center_y - self.hss_y) & 
                         (y <= center_y + self.hss_y), 
                         psi, 0.)

    def derivatives(self, x, y, psi, center_x, center_y):
        """Spatial first derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        psi : 2D array
            Values of the lensing potential at fixed coordinate grid positions.
        center_x : float
            center in x-coordinate
        center_y : float
            center in y-coordinate

        """
        # the following array is at the input (x, y) resolution
        rect_shape = int(np.sqrt(x.size)), int(np.sqrt(y.size))
        pixels = self.function(x, y, psi, center_x, center_y).reshape(rect_shape)
        # we the want to interpolate it to the resolution of the underlying pixelated grid
        # first, get the axes of from input coordinates
        x_coords, y_coords = np.reshape(x, rect_shape)[0, :], np.reshape(y, rect_shape)[:, 0]
        # create the interpolator
        interp = BicubicInterpolator(y_coords, x_coords, pixels)
        # define the 2D coordinate underlying grid
        x_grid, y_grid = np.meshgrid(self.pp.x_coords, self.pp.y_coords)
        # the result is the pixels interpolated on the same grid as the underlying pixelated profile
        pixels_grid = interp(y_grid, x_grid)
        # call the derivatives on the underlying pixelated profile
        return self.pp.derivatives(x, y, pixels_grid)

    def hessian(self, x, y, psi, center_x, center_y):
        """Spatial second derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        psi : 2D array
            Values of the lensing potential at fixed coordinate grid positions.
        center_x : float
            center in x-coordinate
        center_y : float
            center in y-coordinate

        """
        raise NotImplementedError("Computation of Hessian terms for PixelatedPotentialDirac is not implemented.")

    def set_data_pixel_grid(self, pixel_axes):
        self.pp.set_data_pixel_grid(pixel_axes)
        x_coords, y_coords = pixel_axes
        # save half the grid step size in y and y directions
        self.hss_x = np.abs(x_coords[0] - x_coords[1]) / 2.
        self.hss_y = np.abs(y_coords[0] - y_coords[1]) / 2.
