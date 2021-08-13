from jaxtronomy.LensModel.Profiles.base_profile import LensProfileBase
from jaxtronomy.Util.jax_util import BicubicInterpolator


class PixelatedPotential(LensProfileBase):
    param_names = ['pixels']

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
        x_coords : 1D array
            Rectangular x-coordinate grid values.
        y_coords : 1D array
            Rectangular y-coordinate grid values.
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
        x_coords : 1D array
            Rectangular x-coordinate grid values.
        y_coords : 1D array
            Rectangular y-coordinate grid values.
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
        x_coords : 1D array
            Rectangular x-coordinate grid values.
        y_coords : 1D array
            Rectangular y-coordinate grid values.
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
