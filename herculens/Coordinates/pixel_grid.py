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

    def create_model_grid(self, num_pixels=None, pixel_scale_factor=None, grid_center=None, grid_shape=None):
        """
        :param num_pixels: number of pixels on each side. 
        Currently this only works with square grids, so grid_shape needs to setup a square grid.
        :param pixel_scale_factor: multiplicative factor to go from original pixel width to new pixel width.
        If None, defaults to the 1. If num_pixels is provided, pixel_scale_factor is be ignored.
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
        if ((num_pixels is None and (pixel_scale_factor is None or pixel_scale_factor == 1)) or
            (num_pixels is not None and (num_pixels, num_pixels) == self.num_pixel_axes)):
            pixel_scale_factor_ = 1
            num_pixels_ = None
            unchanged_count += 1
        else:
            pixel_scale_factor_ = pixel_scale_factor
            num_pixels_ = num_pixels

        # in case it's the same region as the base coordinate grid
        if unchanged_count == 3:
            return copy.deepcopy(self)

        center_x, center_y = grid_center_
        width, height = grid_shape_
        if num_pixels_ is None:
            pixel_width = self.pixel_width * float(pixel_scale_factor_)
            nx = round(width / pixel_width)
            ny = round(height / pixel_width)
        else:
            nx = ny = num_pixels_  # assuming square grid
            if height != width:
                raise ValueError(f"Setting number of side pixels only works with square grids "
                                 f"(grid shape {grid_shape_} was provided).")
            pixel_width = width / float(nx)

        transform_pix2angle = self.transform_pix2angle / self.pixel_width * pixel_width

        cx, cy = nx / 2., ny / 2.
        cra, cdec = transform_pix2angle.dot(np.array([cx, cy]))
        ra_at_xy_0, dec_at_xy_0 = - cra + center_x + pixel_width/2., - cdec + center_y + pixel_width/2.

        return PixelGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
