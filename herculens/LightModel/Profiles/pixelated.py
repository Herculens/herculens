import jax.numpy as jnp
from herculens.Util.jax_util import BilinearInterpolator, BicubicInterpolator


import numpy as np


class Pixelated(object):
    """Source brightness defined on a fixed coordinate grid."""
    param_names = ['pixels']
    lower_limit_default = {'pixels': -1e10}
    upper_limit_default = {'pixels': 1e10}
    fixed_default = {key: False for key in param_names}
    method_options = ['bilinear', 'bicubic']

    def __init__(self, method='bilinear', allow_extrapolation=True):
        error_msg = "Invalid method. Must be either 'bilinear' or 'bicubic'."
        assert method in self.method_options, error_msg
        if method == 'bilinear':
            self._interp_class = BilinearInterpolator
        else:
            self._interp_class = BicubicInterpolator
        self.data_pixel_area = None
        self.x_coords, self.y_coords = None, None
        self._extrapol_bool = allow_extrapolation

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
        interp = self._interp_class(self.y_coords, self.x_coords, pixels,
                                    allow_extrapolation=self._extrapol_bool)
        # we normalize the interpolated array for correct units when evaluated by LensImage methods
        return interp(y, x) / self.data_pixel_area
    
    def set_data_pixel_grid(self, pixel_axes, data_pixel_area):
        self.data_pixel_area = data_pixel_area
        self.x_coords, self.y_coords = pixel_axes
