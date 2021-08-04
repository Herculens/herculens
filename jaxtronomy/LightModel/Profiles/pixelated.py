import jax.numpy as jnp
from jaxtronomy.Util.jax_util import BilinearInterpolator, BicubicInterpolator


import numpy as np


class Pixelated(object):
    """Source brightness defined on a fixed coordinate grid."""
    param_names = ['pixels']
    method_options = ['bilinear', 'bicubic']

    def __init__(self, x_coords, y_coords, method='bilinear'):
        error_msg = "Invalid method. Must be either 'bilinear' or 'bicubic'."
        assert method in self.method_options, error_msg
        if method == 'bilinear':
            self._interp_class = BilinearInterpolator
        else:
            self._interp_class = BicubicInterpolator
        self._x_coords, self._y_coords = x_coords, y_coords

    def function(self, x, y, pixels):
        """Interpolated evaluation of a pixelated source.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the source.
        x_coords : 1D array
            Rectangular x-coordinate grid values.
        y_coords : 1D array
            Rectangular y-coordinate grid values.
        pixels : 2D array
            Source brightness at fixed coordinate grid positions.
        method : str
            Interpolation method, either 'bilinear' or 'bicubic'.

        """
        # Warning: assuming same pixel size across all the image!
        interp = self._interp_class(self._x_coords, self._y_coords, pixels)
        # we normalize the interpolated array for correct units when evaluated by LensImage methods
        return interp(y, x).T / self._data_pixel_area
    
    def set_data_pixel_area(self, pixel_area):
        self._data_pixel_area = pixel_area
