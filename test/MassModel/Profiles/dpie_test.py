# This file tests the dPIE mass profile against pre-computed maps from GLEE.

import os
import pytest
from astropy.io import fits
import numpy as np

from jax import config
config.update("jax_enable_x64", True)  # could actually make a difference in jifty

import herculens as hcl
from herculens.MassModel.Profiles.dpie import DPIE_GLEE, DPIE_GLEE_STATIC
# from herculens.Util import param_util


GLEE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glee_files')

"""GLEE config file used to generate the reference maps:
dpie
z       0.4000  exact:
0.0  #x-coord   flat:-10,10  step:0.1
0.0  #y-coord   flat:-10,10  step:0.1
0.8  #b/a       flat:0.23,1  step:0.05
0.2  #theta     flat:0,3.14  step:0.05
4    #theta_e   flat:0,100  step:0.1
1    #r_core    flat:0,35  step:0.1
20   #r_trunc   exact:
"""


def get_grid_and_target_maps(glee_scale_flag):
    # Load from the glee output
    if glee_scale_flag is True:
        filename = 'dipe_config_withscale.fits'
    else:
        filename = 'dipe_config_withoutscale.fits'

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

    # npix, npix_y = x_grid.shape
    # assert npix == x_grid.shape[1]
    # pix_scl = abs(x_grid[0, 0] - x_grid[0, 1])
    # assuming x and y are sorted increasingly
    # plt_extent = [x_grid.min()-pix_scl/2., x_grid.max()+pix_scl/2, y_grid.min()-pix_scl/2., y_grid.max()+pix_scl/2]

    kwargs_lens_ref = {
        'theta_E': 4.,
        'r_core': 1.,
        'r_trunc': 20.,
        'q': 0.8, 
        'phi': 0.2,
        'center_x': 0., 
        'center_y': 0.,
    }
    return (
        x_grid, y_grid, 
        alpha_1_ref, alpha_2_ref, 
        kappa_ref, 
        gamma_1_ref, gamma_2_ref, 
        kwargs_lens_ref,
    )

@pytest.mark.parametrize(
    "glee_scale_flag", [False],  # TODO: make test work for flag=True
)
def test_alpha_against_glee(glee_scale_flag):
    (
        x, y,
        alpha_1_ref, alpha_2_ref, _, _, _, 
        kwargs_lens_ref,
    ) = get_grid_and_target_maps(glee_scale_flag)
    profile = DPIE_GLEE(scale_flag=glee_scale_flag)
    alpha_1, alpha_2 = profile.derivatives(
        x, y, **kwargs_lens_ref,
    )
    assert np.allclose(alpha_1, alpha_1_ref, rtol=1e-10)
    assert np.allclose(alpha_2, alpha_2_ref, rtol=1e-10)

@pytest.mark.parametrize(
    "glee_scale_flag", [False],
)
def test_kappa_against_glee(glee_scale_flag):
    (
        x, y,
        _, _, kappa_ref, _, _, 
        kwargs_lens_ref,
    ) = get_grid_and_target_maps(glee_scale_flag)
    profile = DPIE_GLEE(scale_flag=glee_scale_flag)
    f_xx, f_yy, _ = profile.hessian(
        x, y, **kwargs_lens_ref,
    )
    kappa = (f_xx + f_yy) / 2.
    assert np.allclose(kappa, kappa_ref, rtol=1e-6)

# def test_gamma_against_glee():
#     (
#         x, y,
#         _, _, _, gamma_1_ref, gamma_2_ref, 
#         kwargs_lens_ref,
#     ) = get_grid_and_target_maps(glee_scale_flag=False)
#     profile = DPIE_GLEE(scale_flag=False)
#     f_xx, f_yy, f_xy = profile.hessian(
#         x, y, **kwargs_lens_ref,
#     )
#     # gamma_1 = (f_xx - f_yy) / 2.
#     gamma_2 = f_xy
#     # assert np.allclose(gamma_1, gamma_1_ref, rtol=1e-6)
#     assert np.allclose(gamma_2, gamma_2_ref, rtol=1e-6)
