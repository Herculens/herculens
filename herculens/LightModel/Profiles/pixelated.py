# Defines a pixelated profile
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import warnings
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, vmap

from herculens.Util.jax_util import BilinearInterpolator, BicubicInterpolator
from herculens.Util import util


__all__= ['Pixelated']


class Pixelated(object):

    """
    Surface brightness defined on a fixed coordinate grid.
    Note: 'fast_bilinear' is only marginally faster than 'bilinear'.
    """

    param_names = ['pixels']
    lower_limit_default = {'pixels': -1e10}
    upper_limit_default = {'pixels': 1e10}
    fixed_default = {key: False for key in param_names}
    
    _interp_types = ['fast_bilinear', 'bilinear', 'bicubic']
    _deriv_types = ['interpol', 'autodiff']

    def __init__(self, interpolation_type='fast_bilinear', allow_extrapolation=True, derivative_type='autodiff'):
        if interpolation_type not in self._interp_types:
            raise ValueError(f"Invalid method ('{interpolation_type}'). Must be in {self._interp_types}.")
        if derivative_type not in self._deriv_types:
            raise ValueError(f"Unknown derivatives mode '{derivative_type}' "
                             f"(supported types are {self._deriv_types}).")
        self._deriv_type = derivative_type
        self._interp_type = interpolation_type
        self._interp_class = None
        if self._interp_type == 'fast_bilinear':
            try:
                from jaxinterp2d import CartesianGrid
            except ImportError as e:
                warnings.warn("jaxinterp2d is not installed for 'fast_bilinear' option; fall back on the standard 'bilinear' option.")
                self._interp_type = 'bilinear'
            else:
                self._interp_class = CartesianGrid
        if self._interp_class is None:
            if self._interp_type == 'bilinear':
                self._interp_class = BilinearInterpolator
            elif self._interp_type == 'bicubic':
                self._interp_class = BicubicInterpolator
        self._extrapol_bool = allow_extrapolation

        self._data_pixel_area = None
        self._pixel_grid = None
        self._x_coords, self._y_coords = None, None

    @property
    def num_amplitudes(self):
        if self._x_coords is None:
            raise ValueError("No coordinates axes have been set for pixelated profile.")
        return self._x_coords.size

    @property
    def pixel_grid(self):
        return self._pixel_grid

    def function(self, x, y, pixels):
        if self._interp_type == 'fast_bilinear':
            return self._function_fast(x, y, pixels)
        elif self._interp_type in ['bilinear', 'bicubic']:
            return self._function_std(x, y, pixels)

    def _function_fast(self, x, y, pixels):
        """only works when self._interp_type == 'fast_bilinear'"""
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        f = self._interp_class(self._limits, pixels, cval=0.)(y_, x_)
        return f / self._data_pixel_area

    def _function_std(self, x, y, pixels):
        """Interpolated evaluation of a pixelated light profile.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the light profile.
        x_coords : 1D array
            Rectangular x-coordinate grid values.
        y_coords : 1D array
            Rectangular y-coordinate grid values.
        pixels : 2D array
            Surface brightness at fixed coordinate grid positions.
        method : str
            Interpolation method, either 'bilinear' or 'bicubic'.

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        # setup interpolation, assuming cartesian grid
        interp = self._interp_class(self._y_coords, self._x_coords, pixels,
                                    allow_extrapolation=self._extrapol_bool)
        # evaluate the interpolator
        # and normalize for correct units when evaluated by LensImage methods
        f = interp(y_, x_)
        return f / self._data_pixel_area

    def derivatives(self, x, y, pixels):
        if self._deriv_type == 'interpol':
            return self.derivatives_interpol(x, y, pixels)
        elif self._deriv_type == 'autodiff':
            return self.derivatives_autodiff(x, y, pixels)

    def derivatives_interpol(self, x, y, pixels):
        """Spatial first derivatives of the pixelated light profile.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the surface brightness derivatives.
        pixels : 2D array
            Values of the surface brightness at fixed coordinate grid positions (surf. bright. units / arcsec)

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        # setup interpolation, assuming cartesian grid
        interp = self._interp_class(self._y_coords, self._x_coords, pixels)
        # evaluate the interpolator
        f_x = interp(y, x, dy=1) 
        f_y = interp(y, x, dx=1) / self._data_pixel_area
        return f_x / self._data_pixel_area, f_y / self._data_pixel_area

    def derivatives_autodiff(self, x, y, pixels):
        def function(params):
            res = self.function(params[0], params[1], pixels)[0]
            return res
        grad_func = grad(function)
        param_array = jnp.array([x.flatten(), y.flatten()]).T
        res = vmap(grad_func)(param_array)
        f_x = res[:, 0].reshape(*x.shape)
        f_y = res[:, 1].reshape(*x.shape)
        return f_x / self._data_pixel_area, f_y / self._data_pixel_area
    
    def set_pixel_grid(self, pixel_grid, data_pixel_area):
        self._data_pixel_area = data_pixel_area
        self._pixel_grid = pixel_grid
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_grid, y_grid = self.pixel_grid.pixel_coordinates
        nx, ny = x_grid.shape
        x_grid, y_grid = self.pixel_grid.map_coord2pix(util.image2array(x_grid), 
                                                       util.image2array(y_grid))
        self._x_coords = util.array2image(x_grid, nx=nx, ny=ny)[0, :]
        self._y_coords = util.array2image(y_grid, nx=nx, ny=ny)[:, 0]
        self._limits = [
            ( self._y_coords.min(), self._y_coords.max() ), 
            ( self._x_coords.min(), self._x_coords.max() )
        ]
        
