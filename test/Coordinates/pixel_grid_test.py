# Testing modeling workflows
# 
# Copyright (c) 2023, herculens developers and contributors

import pytest
import numpy as np
import numpy.testing as npt

from herculens.Coordinates.pixel_grid import PixelGrid


def _create_pixel_grid(nx, ny, pix_scl, rot_angle, centered=True):
    transform_pix2angle = pix_scl * np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle)], 
            [np.sin(rot_angle),  np.cos(rot_angle)]
        ]
    )
    if centered:
        ra_at_xy_0  = - (nx * pix_scl / 2.) + pix_scl / 2.
        dec_at_xy_0 = - (ny * pix_scl / 2.) + pix_scl / 2.
    else:
        # origin at the "lower-left" pixel
        ra_at_xy_0  = 0.
        dec_at_xy_0 = 0.
    return PixelGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0)


@pytest.mark.parametrize("nx", [10, 11])
@pytest.mark.parametrize("ny", [10, 11])
@pytest.mark.parametrize("pix_scl", [0.1, 1.0])
@pytest.mark.parametrize("rot_angle", [0., 0.5, -0.5])
class TestPixelGrid(object):

    def test_num_pixel(self, nx, ny, pix_scl, rot_angle):
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle)
        assert pixel_grid.num_pixel == nx*ny

    def test_num_pixel_axes(self, nx, ny, pix_scl, rot_angle):
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle)
        assert pixel_grid.num_pixel_axes == (nx, ny)

    def test_width(self, nx, ny, pix_scl, rot_angle):
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle)
        npt.assert_allclose(pixel_grid.width, (nx*pix_scl, ny*pix_scl))

    def test_center(self, nx, ny, pix_scl, rot_angle):
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle, centered=True)
        if rot_angle == 0.:
            expected_cx = 0.
            expected_cy = 0.
            npt.assert_allclose(pixel_grid.center, (expected_cx, expected_cy), atol=1e-10)
        else:
            x_grid, y_grid = pixel_grid.pixel_coordinates
            pixel_grid.center == np.mean(x_grid), np.mean(y_grid)
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle, centered=False)
        if rot_angle == 0.:
            expected_cx = nx/2.*pix_scl - pix_scl/2.
            expected_cy = ny/2.*pix_scl - pix_scl/2.
            npt.assert_allclose(pixel_grid.center, (expected_cx, expected_cy), atol=1e-10)
        else:
            x_grid, y_grid = pixel_grid.pixel_coordinates
            pixel_grid.center == np.mean(x_grid), np.mean(y_grid)

    def test_pixel_width(self, nx, ny, pix_scl, rot_angle):
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle)
        npt.assert_allclose(pixel_grid.pixel_width, pix_scl)

    def test_pixel_area(self, nx, ny, pix_scl, rot_angle):
        pixel_grid = _create_pixel_grid(nx, ny, pix_scl, rot_angle)
        npt.assert_allclose(pixel_grid.pixel_area, pix_scl**2)

