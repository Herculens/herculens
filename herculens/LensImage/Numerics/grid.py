# Handles coordinate grid on which ray-tracing and convolution are performed
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the ImSim.Numerics module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import jax.numpy as np
from herculens.Util import util
from herculens.Util import image_util
from herculens.Coordinates.coord_transforms import Coordinates1D


__all__ = ['RegularGrid']


class RegularGrid(Coordinates1D):
    """
    manages a super-sampled grid on the partial image
    """
    def __init__(self, nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_factor=1):
        """

        :param nx: number of pixels in x-axis
        :param ny: number of pixels in y-axis
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        :param supersampling_indexes: bool array of shape nx x ny, corresponding to pixels being super_sampled
        :param supersampling_factor: int, factor (per axis) of super-sampling
        :param flux_evaluate_indexes: bool array of shape nx x ny, corresponding to pixels being evaluated
        (for both low and high res). Default is None, replaced by setting all pixels to being evaluated.
        """
        super(RegularGrid, self).__init__(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        self._supersampling_factor = supersampling_factor
        self._nx = nx
        self._ny = ny
        self._x_grid, self._y_grid = self.coordinate_grid(nx, ny)
        x_grid_sub, y_grid_sub = util.subgrid_from_coordinate_transform(self._nx, self._nx,
                                                                   transform_pix2angle, ra_at_xy_0, dec_at_xy_0,
                                                                   subgrid_res=self._supersampling_factor)
        self._ra_subgrid = x_grid_sub
        self._dec_subgrid = y_grid_sub

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        return self._ra_subgrid, self._dec_subgrid

    @property
    def grid_points_spacing(self):
        """
        effective spacing between coordinate points, after supersampling
        :return: sqrt(pixel_area)/supersampling_factor
        """
        return self.pixel_width / self._supersampling_factor

    @property
    def num_grid_points_axes(self):
        """
        effective number of points along each axes, after supersampling
        :return: number of pixels per axis, nx*supersampling_factor ny*supersampling_factor
        """
        return self._nx * self._supersampling_factor, self._ny * self._supersampling_factor

    @property
    def num_grid_points(self):
        """
        effective number of points along each axes, after supersampling
        :return: number of pixels per axis, nx*supersampling_factor ny*supersampling_factor
        """
        return self._nx * self._supersampling_factor * self._ny * self._supersampling_factor

    @property
    def supersampling_factor(self):
        """
        :return: factor (per axis) of super-sampling relative to a pixel
        """
        return self._supersampling_factor

    def flux_array2image_low_high(self, flux_array, **kwargs):
        """

        :param flux_array: 1d array of low and high resolution flux values corresponding to the coordinates_evaluate order
        :return: 2d array, 2d array, corresponding to (partial) images in low and high resolution (to be convolved)
        """
        image = self._array2image(flux_array)
        if self._supersampling_factor > 1:
            image_high_res = image
            image_low_res = image_util.re_size(image, self._supersampling_factor)
        else:
            image_high_res = None
            image_low_res = image
        return image_low_res, image_high_res

    def _array2image(self, array):
        """
        maps a 1d array into a (nx, ny) 2d grid with array populating the idex_mask indices

        :param array: 1d array
        :param idex_mask: 1d array of length nx*ny
        :param nx: x-axis of 2d grid
        :param ny: y-axis of 2d grid
        :return:
        """
        nx, ny = self._nx * self._supersampling_factor, self._ny * self._supersampling_factor
        return util.array2image(array, nx, ny)
