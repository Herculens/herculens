# This file tests the dPIE mass profile against pre-computed maps from GLEE.

import os
import pytest
from astropy.io import fits
import numpy as np

from jax import config
config.update("jax_enable_x64", True)  # could actually make a difference in jifty

from herculens.MassModel.Profiles.piemd import PIEMD


GLEE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glee_files', 'piemd')


def get_grid_and_target_maps(glee_scale_flag, index_map):
    # Load from the glee output
    if glee_scale_flag is True:
        filename = f'piemd_config_{index_map}_scale.fits'
    else:
        filename = f'piemd_config_{index_map}_withoutscale.fits'

    # in the FITS file, the HDUs are:
    # x1, x2, alpha1, alpha2, kappa, gamma1, gamma2 (assuming for Dds/Ds=1)
    with fits.open(os.path.join(GLEE_FILE_PATH, filename)) as f:
        glee_cube = f[0].data.astype(np.float64)

    x_grid = glee_cube[0]
    y_grid = glee_cube[1]
    alpha_1_ref = glee_cube[2]
    alpha_2_ref = glee_cube[3]
    kappa_ref = glee_cube[4]
    gamma_1_ref = glee_cube[5]
    gamma_2_ref = glee_cube[6]

    if index_map == 1:
        x_centre = 0.
        y_centre = 0.
        q = 0.8
        pa = 0.2
        theta_E = 4.
        w = 0.2
    elif index_map == 2:
        x_centre = 4.12
        y_centre = 3.90
        q = 0.75
        pa = 3.74
        theta_E = 1.75
        w = 0.001
    elif index_map == 3:
        x_centre = 2.32
        y_centre = 2.1
        q = 0.642
        pa = 2.22222
        theta_E = 3.1
        w = 0.0001
    else:
        raise ValueError("Invalid index_map. Must be 1, 2, or 3.")
        
    kwargs_lens_ref = {
        'theta_E': theta_E,
        'r_core': w,
        'q': q, 
        'phi': pa,
        'center_x': x_centre, 
        'center_y': y_centre,
    }
    return (
        x_grid, y_grid, 
        alpha_1_ref, alpha_2_ref, 
        kappa_ref, 
        gamma_1_ref, gamma_2_ref, 
        kwargs_lens_ref,
    )

@pytest.mark.parametrize(
    "glee_scale_flag", [False, True],
)
@pytest.mark.parametrize(
    "index_map", [1, 2, 3],
)
def test_alpha_against_glee(glee_scale_flag, index_map):
    (
        x, y,
        alpha_1_ref, alpha_2_ref, _, _, _, 
        kwargs_lens_ref,
    ) = get_grid_and_target_maps(glee_scale_flag, index_map)
    profile = PIEMD(r_soft=1e-8, scale_flag=glee_scale_flag)
    alpha_1, alpha_2 = profile.derivatives(
        x, y, **kwargs_lens_ref,
    )
    assert np.allclose(alpha_1, alpha_1_ref, rtol=1e-10)
    assert np.allclose(alpha_2, alpha_2_ref, rtol=1e-10)

@pytest.mark.parametrize(
    "glee_scale_flag", [False, True],
)
@pytest.mark.parametrize(
    "index_map", [1, 2, 3],
)
def test_kappa_against_glee(glee_scale_flag, index_map):
    (
        x, y,
        _, _, kappa_ref, _, _, 
        kwargs_lens_ref,
    ) = get_grid_and_target_maps(glee_scale_flag, index_map)
    profile = PIEMD(r_soft=1e-8, scale_flag=glee_scale_flag)
    f_xx, f_yy, _ = profile.hessian(
        x, y, **kwargs_lens_ref,
    )
    kappa = (f_xx + f_yy) / 2.
    assert np.allclose(kappa, kappa_ref, rtol=1e-6)

# @pytest.mark.parametrize(
#     "glee_scale_flag", [False, True],
# )
# @pytest.mark.parametrize(
#     "index_map", [1, 2, 3],
# )
# def test_gamma_against_glee(glee_scale_flag, index_map):
#     (
#         x, y,
#         _, _, _, gamma_1_ref, gamma_2_ref, 
#         kwargs_lens_ref,
#     ) = get_grid_and_target_maps(glee_scale_flag, index_map)
#     profile = PIEMD(r_soft=1e-8, scale_flag=glee_scale_flag)
#     f_xx, f_yy, f_xy = profile.hessian(
#         x, y, **kwargs_lens_ref,
#     )
#     gamma_1 = (f_xx - f_yy) / 2.
#     gamma_2 = f_xy
#     assert np.allclose(gamma_1, gamma_1_ref, rtol=1e-6)
#     assert np.allclose(gamma_2, gamma_2_ref, rtol=1e-6)