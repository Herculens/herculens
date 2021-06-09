from jaxtronomy.Util.jax_util import BilinearInterpolator, BicubicInterpolator


class PixelatedSource(object):
    """Source brightness defined on a fixed coordinate grid."""
    param_names = ['x_coords', 'y_coords', 'image']

    def __init__(self):
        self.method_options = ['bilinear', 'bicubic']

    def function(self, x, y, x_coords, y_coords, image, method='bilinear'):
        """Interpolated evaluation of a pixelated source.

        Parameters
        ----------
        x, y : array-like
            Coordinates at which to evaluate the source.
        x_coords : 1D array
            Rectangular x-coordinate grid values.
        y_coords : 1D array
            Rectangular y-coordinate grid values.
        image : 2D array
            Source brightness at fixed coordinate grid positions.
        method : str
            Interpolation method, either 'bilinear' or 'bicubic'.

        """
        error_msg = "Invalid method. Must be either 'bilinear' or 'bicubic'."
        assert method in self.method_options, error_msg
        if method == 'bilinear':
            interp = BilinearInterpolator(x_coords, y_coords, image)
        else:
            interp = BicubicInterpolator(x_coords, y_coords, image)
        return interp(x, y).T
