import pytest
import numpy as np
import numpy.testing as npt
import copy
import unittest

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Util import util


class TestData(object):
    def setup(self):
        self.numPix = 100
        pix_scl = 0.08  # arcsec / pixel
        half_size = self.numPix * pix_scl / 2
        ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2
        transform_pix2angle = pix_scl * np.eye(2)
        kwargs_grid = {'nx': self.numPix, 'ny': self.numPix,
                        'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                        'transform_pix2angle': transform_pix2angle}
        self.Grid = PixelGrid(**kwargs_grid)

    def test_numData(self):
        assert self.Grid.num_pixel == self.numPix ** 2

    def test_shift_coordinate_system(self):
        x_shift = 0.05
        y_shift = 0

        numPix = 100
        pix_scl = 0.08  # arcsec / pixel
        half_size = numPix * pix_scl / 2
        ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2
        transform_pix2angle = pix_scl * np.eye(2)

        kwargs_grid = {'nx': numPix, 'ny': numPix, 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                       'transform_pix2angle': transform_pix2angle}

        data = PixelGrid(**kwargs_grid)
        data_new = copy.deepcopy(data)
        data_new.shift_coordinate_system(x_shift, y_shift, pixel_unit=False)
        ra, dec = 0, 0
        x, y = data.map_coord2pix(ra, dec)
        x_new, y_new = data_new.map_coord2pix(ra + x_shift, dec + y_shift)
        npt.assert_almost_equal(x, x_new, decimal=10)
        npt.assert_almost_equal(y, y_new, decimal=10)

        ra, dec = data.map_pix2coord(x, y)
        ra_new, dec_new = data_new.map_pix2coord(x, y)
        npt.assert_almost_equal(ra, ra_new-x_shift, decimal=10)
        npt.assert_almost_equal(dec, dec_new-y_shift, decimal=10)

        x_coords, y_coords = data.pixel_coordinates
        x_coords_new, y_coords_new = data_new.pixel_coordinates
        npt.assert_almost_equal(x_coords[0], x_coords_new[0]-x_shift, decimal=10)
        npt.assert_almost_equal(y_coords[0], y_coords_new[0]-y_shift, decimal=10)


if __name__ == '__main__':
    pytest.main()
