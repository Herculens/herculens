# Defines a pixelated profile in the lens potential
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, vmap
# from functools import partial

from utax.interpolation import BilinearInterpolator, BicubicInterpolator

from herculens.Util import util


__all__ = ['PixelatedPotential', 'PixelatedPotentialDirac', 'PixelatedFixed']


class PixelatedPotential(object):
    param_names = ['pixels']
    lower_limit_default = {'pixels': -1e10}
    upper_limit_default = {'pixels': 1e10}
    fixed_default = {key: False for key in param_names}

    _interp_types = ['fast_bilinear', 'bicubic']
    _deriv_types = ['interpol', 'autodiff']

    def __init__(self, interpolation_type='fast_bilinear', derivative_type='autodiff'):
        """Lensing potential on a fixed coordinate grid."""
        print("CHECK", interpolation_type, derivative_type)
        if interpolation_type not in self._interp_types:
            raise ValueError(f"Invalid interpolated mode ('{interpolation_type}'). Must be in {self._interp_types}.")
        if derivative_type not in self._deriv_types:
            raise ValueError(f"Unknown derivatives mode '{derivative_type}' "
                             f"(supported types are {self._deriv_types}).")
        self._deriv_type = derivative_type
        self._interp_class = None
        self._interp_type = interpolation_type
        if self._interp_type == 'fast_bilinear':
            try:
                from jaxinterp2d import CartesianGrid
            except ImportError as e:
                warnings.warn("jaxinterp2d is not installed for 'fast_bilinear' option; fall back on the standard 'bicubic' option.")
                self._interp_type = 'bicubic'
            else:
                self._interp_class = CartesianGrid
        if self._interp_class is None:
            if self._interp_type == 'bicubic':
                from utax.interpolation import BicubicInterpolator
                self._interp_class = BicubicInterpolator
        
        self._pixel_grid = None
        self._x_coords, self._y_coords = None, None

    @property
    def pixel_grid(self):
        return self._pixel_grid

    def function(self, x, y, pixels=None):
        if self._interp_type == 'fast_bilinear':
            return self._function_fast(x, y, pixels)
        elif self._interp_type == 'bicubic':
            return self._function_std(x, y, pixels)

    def _function_fast(self, x, y, pixels):
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        interp = self._interp_class(self._limits, pixels, cval=0.)
        return interp(y_, x_)

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
        # Due to matching scipy's interpolation, we need to switch x and y
        # coordinates as well as transpose
        interp = self._interp_class(self._y_coords, self._x_coords, pixels)
        # evaluate the interpolator
        return interp(y_, x_)

    def derivatives(self, x, y, pixels):
        if self._deriv_type == 'interpol':
            return self.derivatives_interpol(x, y, pixels)
        elif self._deriv_type == 'autodiff':
            return self.derivatives_autodiff(x, y, pixels)

    def derivatives_interpol(self, x, y, pixels):
        """Spatial first derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        interp = BicubicInterpolator(self._y_coords, self._x_coords, pixels)
        f_x = interp(y_, x_, dy=1)
        f_y = interp(y_, x_, dx=1)
        return f_x, f_y

    def derivatives_autodiff(self, x, y, pixels):
        def function(params):
            res = self.function(params[0], params[1], pixels) #[0]
            return res
        grad_func = grad(function)
        param_array = jnp.array([x.flatten(), y.flatten()]).T
        res = vmap(grad_func)(param_array)
        f_x = res[:, 0].reshape(*x.shape)
        f_y = res[:, 1].reshape(*x.shape)
        return f_x, f_y

    def hessian(self, x, y, pixels):
        if self._deriv_type == 'interpol':
            return self.hessian_interpol(x, y, pixels)
        elif self._deriv_type == 'autodiff':
            return self.hessian_autodiff(x, y, pixels)

    def hessian_interpol(self, x, y, pixels):
        """Spatial second derivatives of the lensing potential.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the lensing potential derivatives.
        pixels : 2D array
            Values of the lensing potential at fixed coordinate grid positions.

        """
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = self.pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        interp = BicubicInterpolator(self._y_coords, self._x_coords, pixels)
        # TODO Why doesn't this follow the pattern of the first derivatives ?
        f_xx = interp(y_, x_, dx=2)
        f_yy = interp(y_, x_, dy=2)
        f_xy = interp(y_, x_, dx=1, dy=1)
        return f_xx, f_yy, f_xy

    def hessian_autodiff(self, x, y, pixels):
        def function(params):
            res = self.function(params[0], params[1], pixels)[0]
            return res
        hessian_func = jacfwd(jacrev(function))
        param_array = jnp.array([x.flatten(), y.flatten()]).T
        res = vmap(hessian_func)(param_array)
        f_xx = res[:, 0, 0].reshape(*x.shape)
        f_xy = res[:, 0, 1].reshape(*x.shape)
        # f_yx = res[:, 1, 0].reshape(*x.shape)
        f_yy = res[:, 1, 1].reshape(*x.shape)
        return f_xx, f_yy, f_xy

    def set_pixel_grid(self, pixel_grid):
        self._pixel_grid = pixel_grid
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_grid, y_grid = self.pixel_grid.pixel_coordinates
        x_grid, y_grid = self.pixel_grid.map_coord2pix(util.image2array(x_grid), util.image2array(y_grid))
        x_grid, y_grid = util.array2image(x_grid), util.array2image(y_grid)
        self._x_coords = x_grid[0, :]
        self._y_coords = y_grid[:, 0]
        self._limits = [
            ( self._y_coords.min(), self._y_coords.max() ), 
            ( self._x_coords.min(), self._x_coords.max() )
        ]



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



class PixelatedFixed(object):
    # param_names = ['pot_pixels',
    #                'deriv_pixels',
    #                'hess_pixels']
    # lower_limit_default = {'pixels': -1e10}
    # upper_limit_default = {'pixels': 1e10}
    # fixed_default = {key: False for key in param_names}
    def __init__(self, func_pixel_grid=None, deriv_pixel_grid=None, hess_pixel_grid=None):
        """
        Potential and its spatial derivatives defined on a pixelated grid (can be different for each of these).
        IMPORTANT: This profile is NOT designed to be used during optimization!!
        """
        self._func_pixel_grid  = func_pixel_grid
        self._deriv_pixel_grid = deriv_pixel_grid
        self._hess_pixel_grid  = hess_pixel_grid
        try:
            from jaxinterp2d import CartesianGrid
        except ImportError as e:
            raise NotImplementedError("`jaxinterp2d` is not installed and "
                                      "the use of the bilinear interpolator from `utax` is not implemented.")
            # self._interp_type = 'bilinear'
            # self._interp_class = BilinearInterpolator
        else:
            self._interp_type = 'fast_bilinear'
            self._interp_class = CartesianGrid

    def get_pixel_grid(self, component='func_pixels'):
        if component == 'func_pixels':
            return self._func_pixel_grid
        elif component == 'deriv_pixels':
            return self._deriv_pixel_grid
        elif component == 'hess_pixels':
            return self._hess_pixel_grid
        else:
            raise ValueError(f"Invalid component '{component}' for pixelated mass profile.")

    def function(self, x, y, func_pixels=None, deriv_pixels=None, hess_pixels=None):
        return self._interpol(x, y, func_pixels, self._func_pixel_grid)
    
    def derivatives(self, x, y, func_pixels=None, deriv_pixels=None, hess_pixels=None):
        deriv_x, deriv_y = deriv_pixels
        return (self._interpol(x, y, deriv_x, self._deriv_pixel_grid), 
                self._interpol(x, y, deriv_y, self._deriv_pixel_grid))

    def hessian(self, x, y, func_pixels=None, deriv_pixels=None, hess_pixels=None):
        hess_xx, hess_yy, hess_xy = hess_pixels
        return (self._interpol(x, y, hess_xx, self._hess_pixel_grid), 
                self._interpol(x, y, hess_yy, self._hess_pixel_grid),
                self._interpol(x, y, hess_xy, self._hess_pixel_grid))

    def _interpol(self, x, y, pixels, pixel_grid):
        # ensure the coordinates are cartesian by converting angular to pixel units
        x_, y_ = pixel_grid.map_coord2pix(x.flatten(), y.flatten())
        x_, y_ = x_.reshape(*x.shape), y_.reshape(*y.shape)
        x_coord, y_coord = pixel_grid.pixel_axes
        x_coord_pix, y_coord_pix = pixel_grid.map_coord2pix(x_coord, y_coord)
        limits = [
            ( y_coord_pix.min(), y_coord_pix.max() ), 
            ( x_coord_pix.min(), x_coord_pix.max() )
        ]
        interp = self._interp_class(limits, pixels, cval=0.)
        return interp(y_, x_)
