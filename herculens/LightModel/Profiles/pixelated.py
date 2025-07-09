# Defines a pixelated profile
# 
# Copyright (c) 2023, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import warnings
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, vmap

from utax.interpolation import BilinearInterpolator, BicubicInterpolator

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

    def __init__(self, interpolation_type='fast_bilinear', allow_extrapolation=True, 
                 derivative_type='interpol', adaptive_grid=False):
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
        self._adaptive_grid = adaptive_grid

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
    
    @property
    def is_adaptive(self):
        return self._adaptive_grid

    def function(self, x, y, pixels_x_coord=None, pixels_y_coord=None, pixels=None):
        if self._interp_type == 'fast_bilinear':
            f = self._function_fast(x, y, pixels_x_coord, pixels_y_coord, pixels)
        elif self._interp_type in ['bilinear', 'bicubic']:
            f = self._function_std(x, y, pixels_x_coord, pixels_y_coord, pixels)
        # normalize for correct units when evaluated by LensImage methods
        return f / self._data_pixel_area

    def _function_fast(self, x, y, pixels_x_coord, pixels_y_coord, pixels):
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        if self._adaptive_grid:
            x_coord_pix, y_coord_pix = self.pixel_grid.map_coord2pix(pixels_x_coord, pixels_y_coord)
            limits = [
                ( y_coord_pix.min(), y_coord_pix.max() ), 
                ( x_coord_pix.min(), x_coord_pix.max() )
            ]
        else:
            limits = self._limits
        interp = self._interp_class(limits, pixels, cval=0.)
        f = interp(y_, x_)
        return f

    def _function_std(self, x, y, pixels_x_coord, pixels_y_coord, pixels):
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
        if not self._adaptive_grid:  # in this case pixels_x_coord and pixels_y_coord should be None
            pixels_x_coord, pixels_y_coord = self._x_coords, self._y_coords
        interp = self._interp_class(pixels_y_coord, pixels_x_coord, pixels,
                                    allow_extrapolation=self._extrapol_bool)
        # evaluate the interpolator
        f = interp(y_, x_)
        return f

    def derivatives(self, x, y, pixels_x_coord=None, pixels_y_coord=None, pixels=None):
        """Spatial first derivatives of the pixelated light profile.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the surface brightness derivatives.
        pixels : 2D array
            Values of the surface brightness at fixed coordinate grid positions (surf. bright. units / arcsec)

        """
        if self._deriv_type == 'interpol':
            f_x, f_y = self._derivatives_interpol(x, y, pixels_x_coord, pixels_y_coord, pixels)
        elif self._deriv_type == 'autodiff':
            f_x, f_y = self._derivatives_autodiff(x, y, pixels_x_coord, pixels_y_coord, pixels)
        # normalize for correct units when evaluated by LensImage methods
        return f_x / self._data_pixel_area, f_y / self._data_pixel_area

    def _derivatives_interpol(self, x, y, pixels_x_coord, pixels_y_coord, pixels):
        if self._interp_type not in ['bilinear', 'bicubic']:
            raise ValueError(f"Invalid interpolation type '{self._interp_type}' for "
                             f"'interpolated' derivatives computation. "
                             f"Either use 'autodiff' derivatives computation, "
                             f"or choose interpolation type 'bilinear' or 'bicubic'.")
        elif self._interp_type == 'bilinear':
            interp_class = BilinearInterpolator
        else:
            interp_class = BicubicInterpolator
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        # setup interpolation, assuming cartesian grid
        if not self._adaptive_grid:  # in this case pixels_x_coord and pixels_y_coord should be None
            pixels_x_coord, pixels_y_coord = self._x_coords, self._y_coords
        interp = interp_class(pixels_y_coord, pixels_x_coord, pixels,
                              allow_extrapolation=self._extrapol_bool)
        # evaluate the interpolator
        f_x = interp(y_, x_, dy=1) / self._delta_x
        f_y = interp(y_, x_, dx=1) / self._delta_y
        return f_x, f_y

    def _derivatives_autodiff(self, x, y, pixels_x_coord, pixels_y_coord, pixels):
        def function(params):
            res = self.function(
                params[0], params[1], 
                pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord,
                pixels=pixels)
            if self._interp_type != 'fast_bilinear':
                res = res[0]
            return res
        grad_func = grad(function)
        param_array = jnp.array([x.flatten(), y.flatten()]).T
        res = vmap(grad_func)(param_array)
        f_x = res[:, 0].reshape(*x.shape)
        f_y = res[:, 1].reshape(*x.shape)
        return f_x, f_y
    
    def set_pixel_grid(self, pixel_grid, data_pixel_area):
        self._data_pixel_area = data_pixel_area
        self._pixel_grid = pixel_grid
        # compute the original pixel size (used for proper scaling of derivatives)
        x_grid, y_grid = self.pixel_grid.pixel_coordinates
        self._delta_x = abs(x_grid[0, 1] - x_grid[0, 0])
        self._delta_y = abs(y_grid[1, 0] - y_grid[0, 0])
        # ensure the coordinates are cartesian by converting angular to pixel units
        nx, ny = x_grid.shape
        x_grid, y_grid = self.pixel_grid.map_coord2pix(util.image2array(x_grid), 
                                                       util.image2array(y_grid))
        self._x_coords = util.array2image(x_grid, nx=nx, ny=ny)[0, :]
        self._y_coords = util.array2image(y_grid, nx=nx, ny=ny)[:, 0]
        self._limits = [
            ( self._y_coords.min(), self._y_coords.max() ), 
            ( self._x_coords.min(), self._x_coords.max() )
        ]
        
