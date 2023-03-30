# Testing modeling workflows
# 
# Copyright (c) 2023, herculens developers and contributors

import pytest
import numpy as np

from herculens.Coordinates.coord_transforms import Coordinates


@pytest.mark.parametrize(
    "pix_scl",
    [0.05, 1.0]
)
def test_shift_coordinate_system(pix_scl):
    transform_pix2angle = np.array([[pix_scl, 0], [0, pix_scl]])
    ra_0, dec_0 = 1., 1.
    coords = Coordinates(
        transform_pix2angle=transform_pix2angle,
        ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
    )
    x0, y0 = coords.xy_at_radec_0
    coords.shift_coordinate_system(x_shift=pix_scl, y_shift=0, pixel_unit=False)
    x0_new, y0_new = coords.xy_at_radec_0
    assert x0_new == x0 - 1.
    coords.shift_coordinate_system(x_shift=0, y_shift=pix_scl, pixel_unit=False)
    x0_new, y0_new = coords.xy_at_radec_0
    assert y0_new == y0 - 1. and x0_new == x0 - 1.

    coords = Coordinates(
        transform_pix2angle=transform_pix2angle,
        ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
    )
    x0, y0 = coords.xy_at_radec_0
    coords.shift_coordinate_system(x_shift=1, y_shift=0, pixel_unit=True)
    x0_new, y0_new = coords.xy_at_radec_0
    assert x0_new == x0 - 1.
    coords.shift_coordinate_system(x_shift=0, y_shift=1, pixel_unit=True)
    x0_new, y0_new = coords.xy_at_radec_0
    assert y0_new == y0 - 1. and x0_new == x0 - 1.
