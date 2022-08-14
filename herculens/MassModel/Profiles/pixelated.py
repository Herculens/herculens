# Defines a pixelated profile in the lens potential
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

from herculens.Util.jax_util import BicubicInterpolator
from herculens.Util import util


__all__ = ['PixelatedPotential', 'PixelatedPotentialDirac']


class PixelatedPotential(object):
    param_names = ['pixels']
    lower_limit_default = {'pixels': -1e10}
    upper_limit_default = {'pixels': 1e10}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        """Lensing potential on a fixed coordinate grid."""
        super(PixelatedPotential, self).__init__()
        self._pixel_grid = None
        self._x_coords, self._y_coords = None, None

    @property
    def pixel_grid(self):
        return self._pixel_grid

    def function(self, x, y, pixels):
        """Interpolated evaluation of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x, y)
        # Due to matching scipy's interpolation, we need to switch x and y
        # coordinates as well as transpose
        interp = BicubicInterpolator(self._y_coords, self._x_coords, pixels)
        return interp(y_, x_)

    def derivatives(self, x, y, pixels):
        """Spatial first derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x, y)
        interp = BicubicInterpolator(self._y_coords, self._x_coords, pixels)
        return interp(y_, x_, dy=1), interp(y_, x_, dx=1)

    def hessian(self, x, y, pixels):
        """Spatial second derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x, y)
        interp = BicubicInterpolator(self._y_coords, self._x_coords, pixels)
        # TODO Why doesn't this follow the pattern of the first derivatives ?
        psi_xx = interp(y_, x_, dx=2)
        psi_yy = interp(y_, x_, dy=2)
        psi_xy = interp(y_, x_, dx=1, dy=1)
        return psi_xx, psi_yy, psi_xy

    def set_pixel_grid(self, pixel_grid):
        self._pixel_grid = pixel_grid
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_grid, y_grid = self.pixel_grid.pixel_coordinates
        x_grid, y_grid = self.pixel_grid.map_coord2pix(util.image2array(x_grid), 
                                                       util.image2array(y_grid))
        self._x_coords = util.array2image(x_grid)[0, :]
        self._y_coords = util.array2image(y_grid)[:, 0]



class PixelatedPotentialDirac(object):
    param_names = ['psi', 'center_x', 'center_y']
    lower_limit_default = {'psi': -1e10, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'psi': 1e10, 'center_x': 100, 'center_y': 100}
    fixed_default = {key: False for key in param_names}

    def __init__(self):
        """Dirac impulse in potential on a fixed coordinate grid."""
        super(PixelatedPotentialDirac, self).__init__()
        self.pp = PixelatedPotential()

    @property
    def pixel_grid(self):
        return self.pp.pixel_grid

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
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x, y)
        return jnp.where((x_ >= center_x - self.hss_x) & 
                         (x_ <= center_x + self.hss_x) & 
                         (y_ >= center_y - self.hss_y) & 
                         (y_ <= center_y + self.hss_y), 
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
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x, y)
        # the following array is at the input (x, y) resolution
        rect_shape = int(np.sqrt(x_.size)), int(np.sqrt(y_.size))
        pixels = self.function(x_, y_, psi, center_x, center_y).reshape(rect_shape)
        # we the want to interpolate it to the resolution of the underlying pixelated grid
        # first, get the axes of from input coordinates
        x_coords, y_coords = np.reshape(x_, rect_shape)[0, :], np.reshape(y_, rect_shape)[:, 0]
        # create the interpolator
        interp = BicubicInterpolator(y_coords, x_coords, pixels)
        # define the 2D coordinate underlying grid
        x_grid, y_grid = np.meshgrid(self.pp.x_coords, self.pp.y_coords)
        # the result is the pixels interpolated on the same grid as the underlying pixelated profile
        pixels_grid = interp(y_grid, x_grid)
        # call the derivatives on the underlying pixelated profile
        return self.pp.derivatives(x_, y_, pixels_grid)

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

    def set_pixel_grid(self, pixel_grid):
        self.pp.set_pixel_grid(pixel_grid)
        self.hss_x = np.abs(self.pp._x_coords[0] - self.pp._x_coords[1]) / 2.
        self.hss_y = np.abs(self.pp._y_coords[0] - self.pp._y_coords[1]) / 2.
