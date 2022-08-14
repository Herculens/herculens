# Handles data and model pixelated grids
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Data module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


# TODO: combine with grid class from numerics

import copy
import numpy as np
from herculens.Coordinates.coord_transforms import Coordinates
from herculens.Util import util


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

    # def model_pixel_coordinates(self, name):
    #     """

    #     :return: RA coords, DEC coords
    #     """
    #     if self._model_grids[name] is None:
    #         return None
    #     return self._model_grids[name][0], self._model_grids[name][1]

    # def model_pixel_axes(self, name, pixel_unit=False):
    #     """

    #     :return: RA coords, DEC coords
    #     """
    #     if self._model_grids[name] is None:
    #         return None
    #     x_grid, y_grid = self.model_pixel_coordinates(name)
    #     if pixel_unit is True:
    #         x_grid, y_grid = self.map_coord2pix(name)
    #     return x_grid[0, :], y_grid[:, 0]

    # def model_pixel_extent(self, name):
    #     """

    #     :return: RA coords, DEC coords
    #     """
    #     if self._model_grids[name] is None:
    #         return None
    #     x_coords, y_coords = self.model_pixel_axes(name)
    #     return [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # def model_pixel_shape(self, name):
    #     """

    #     :return: RA coords, DEC coords
    #     """
    #     if self._model_grids[name] is None:
    #         return None
    #     return self._model_grids[name][0].shape

    # def model_pixel_width(self, name):
    #     """

    #     :return: RA coords, DEC coords
    #     """
    #     if self._model_grids[name] is None:
    #         return None
    #     extent = self.model_pixel_extent(name)
    #     x_coords, y_coords = self.model_pixel_axes(name)
    #     pix_width_x = np.abs(x_coords[1] - x_coords[0])
    #     pix_width_y = np.abs(y_coords[1] - y_coords[0])
    #     return np.sqrt(pix_width_x * pix_width_y)

    # def remove_model_grid(self, name):
    #     if name in self._model_grids:
    #         del self._model_grids[name]

    def create_model_grid(self, pixel_scale_factor=None, grid_center=None, grid_shape=None):
        """
        :param pixel_scale_factor: multiplicative factor to go from original pixel width to new pixel width.
        If None, defaults to the 1.
        :param grid_center: 2-tuple (center_x, center_y) with grid center in angular units
        If None, defaults to the original grid center. 
        :param grid_shape: 2-tuple (width, height) window size in angular units
        If None, defaults to the original window size.
        """
        unchanged_count = 0
        if grid_center is None or grid_center == self.center:
            grid_center_ = self.center
            unchanged_count += 1
        else:
            grid_center_ = grid_center
        if grid_shape is None or grid_shape == self.width:
            grid_shape_ = self.width
            unchanged_count += 1
        else:
            grid_shape_ = grid_shape
        if pixel_scale_factor is None or pixel_scale_factor == 1:
            pixel_scale_factor_ = 1
            unchanged_count += 1
        else:
            pixel_scale_factor_ = pixel_scale_factor

        # in case it's the same region as the base coordinate grid
        if unchanged_count == 3:
            return copy.deepcopy(self)

        pixel_width = self.pixel_width * float(pixel_scale_factor_)
        center_x, center_y = grid_center_
        width, height = grid_shape_
        nx = round(width / pixel_width)
        ny = round(height / pixel_width)

        transform_pix2angle = self.transform_pix2angle / self.pixel_width * pixel_width

        cx, cy = int(nx / 2), int(ny / 2)
        # if nx % 2 == 0:
        #     cx += 0.5  # makes sure the center is in the middle of a pixel
        # if ny % 2 == 0:
        #     cy += 0.5  # makes sure the center is in the middle of a pixel

        cra, cdec = transform_pix2angle.dot(np.array([cx, cy]))
        ra_at_xy_0, dec_at_xy_0 = - cra + center_x, - cdec + center_y

        return PixelGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0)

    # def create_model_grid_simple(self, original_shape, original_extent, name='none', overwrite=False):
    #     """
    #     Simpler version typically for MOLET grids.
    #     """
    #     if not overwrite and name in self._model_grids:
    #         return
    #     nx, ny = original_shape
    #     width = original_extent[1] - original_extent[0]
    #     height = original_extent[3] - original_extent[2]
    #     pixel_width = np.sqrt((width / nx) * (height / ny))
    #     #semi_width  = (width - pixel_width)/2.
    #     #semi_height = (height - pixel_width)/2.
    #     extent = [
    #         original_extent[0] + pixel_width/2., original_extent[1] - pixel_width/2.,
    #         original_extent[2] + pixel_width/2., original_extent[3] - pixel_width/2.
    #     ]
    #     x_coords = np.linspace(extent[0], extent[1], nx) # * x_sign
    #     y_coords = np.linspace(extent[2], extent[3], ny) # * y_sign
    #     x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    #     self._model_grids[name] = (x_grid, y_grid)

    # def create_model_grid_old(self, factor, name='none', mode='supersampling'):
    #     if factor is None:
    #         # avoid unnecessary computations
    #         self._model_grids[name] = None
    #         return
    #     if factor < 1:
    #         raise ValueError(f"{mode}-sampling factor must be equal to or greater than 1")
    #     if factor == 1:
    #         x_grid = np.copy(self._x_grid)
    #         y_grid = np.copy(self._y_grid)
    #     else:
    #         if mode == 'supersampling':
    #             nx = int(self._nx * factor)
    #             ny = int(self._ny * factor)
    #         elif mode == 'undersampling':
    #             nx = int(self._nx / factor)
    #             ny = int(self._ny / factor)
    #         else:
    #             raise ValueError(f"Mode '{mode}' for creating new coordinate grid is not supported")
    #         extent = self.extent
    #         x_coords = np.linspace(extent[0], extent[1], nx)
    #         y_coords = np.linspace(extent[2], extent[3], ny)
    #         x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    #     self._model_grids[name] = (x_grid, y_grid)
