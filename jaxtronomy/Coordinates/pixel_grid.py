import numpy as np
from jaxtronomy.Coordinates.coord_transforms import Coordinates

__all__ = ['PixelGrid']


class PixelGrid(Coordinates):
    """
    class that manages a specified pixel grid (rectangular at the moment) and its coordinates
    """

    def __init__(self, nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0):
        """

        :param nx: number of pixels in x-axis
        :param ny: number of pixels in y-axis
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        """
        super(PixelGrid, self).__init__(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        self._nx = nx
        self._ny = ny
        self._x_grid, self._y_grid = self.coordinate_grid(nx, ny)
        self._model_grids = {}

    @property
    def num_pixel(self):
        """

        :return: number of pixels in the data
        """
        return self._nx * self._ny

    @property
    def num_pixel_axes(self):
        """

        :return: number of pixels per axis, nx ny
        """
        return self._nx, self._ny

    @property
    def width(self):
        """

        :return: width of data frame
        """
        return self._nx * self.pixel_width, self._ny * self.pixel_width

    @property
    def center(self):
        """

        :return: center_x, center_y of coordinate system
        """
        return np.mean(self._x_grid), np.mean(self._y_grid)

    def shift_coordinate_system(self, x_shift, y_shift, pixel_unit=False):
        """
        shifts the coordinate system
        :param x_shift: shift in x (or RA)
        :param y_shift: shift in y (or DEC)
        :param pixel_unit: bool, if True, units of pixels in input, otherwise RA/DEC
        :return: updated data class with change in coordinate system
        """
        self._shift_coordinates(x_shift, y_shift, pixel_unit=pixel_unit)
        self._x_grid, self._y_grid = self.coordinate_grid(self._nx, self._ny)

    @property
    def pixel_coordinates(self):
        """

        :return: RA coords, DEC coords
        """
        return self._x_grid, self._y_grid

    @property
    def pixel_axes(self):
        """

        :return: RA coords, DEC coords
        """
        return self._x_grid[0, :], self._y_grid[:, 0]

    @property
    def extent(self):
        x_coords, y_coords = self.pixel_axes
        return [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    def model_pixel_coordinates(self, name):
        """

        :return: RA coords, DEC coords
        """
        if self._model_grids[name] is None:
            return None
        return self._model_grids[name][0], self._model_grids[name][1]

    def model_pixel_axes(self, name):
        """

        :return: RA coords, DEC coords
        """
        if self._model_grids[name] is None:
            return None
        return self._model_grids[name][2], self._model_grids[name][3]

    def create_model_grid(self, factor, name='none', mode='supersampling'):
        if factor is None:
            # avoid unnecessary computations
            self._model_grids[name] = None
            return
        if factor < 1:
            raise ValueError(f"{mode}-sampling factor must be equal to or greater than 1")
        if factor == 1:
            x_grid = np.copy(self._x_grid)
            y_grid = np.copy(self._y_grid)
            x_coords, y_coords = x_grid[0, :], y_grid[:, 0]
        else:
            if mode == 'supersampling':
                nx = self._nx * int(factor)
                ny = self._ny * int(factor)
            else:
                nx = self._nx // int(factor)
                ny = self._ny // int(factor)
            extent = self.extent
            x_coords = np.linspace(extent[0], extent[1], nx)
            y_coords = np.linspace(extent[2], extent[3], ny)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        self._model_grids[name] = (x_grid, y_grid, x_coords, y_coords)
