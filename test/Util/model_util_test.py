# This file provides unit tests for functions in herculens.Util.model_util.py.

import pytest
import numpy as np

import herculens as hcl
import herculens.Util.model_util as mut


# Create pytest ficture to define a LensImage instance
@pytest.fixture
def lens_image():
    # Create an instance of the LensImage class
    lens_image = hcl.LensImage(
        hcl.PixelGrid(nx=8, ny=8),
        hcl.PSF(psf_type='GAUSSIAN', fwhm=0.2, pixel_size=0.1),
        noise_class=hcl.Noise(nx=8, ny=8, noise_map=np.ones((8, 8))),
        lens_mass_model_class=hcl.MassModel([hcl.EPL(), hcl.Shear()]),
        source_model_class=hcl.LightModel(
            hcl.PixelatedLight(), 
            kwargs_pixelated={'num_pixels': 12},
        ),
    )
    return lens_image

@pytest.fixture
def kwargs_model():
    # Returns the full dictionary of model parameters consistent 
    # with the LensImage model defined above
    return {
        'kwargs_lens': [
            {'theta_E': 1.0, 'gamma': 2., 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0},
            {'gamma1': 0.05, 'gamma2': 0.0},
        ],
        'kwargs_source': [
            {'pixels': np.random.randn(12, 12)}
        ],
    }

def test_critical_lines_caustics(lens_image, kwargs_model):
    # Call the tested function
    critical_lines, caustics, centers = mut.critical_lines_caustics(
        lens_image, 
        kwargs_model['kwargs_lens'],
        return_lens_centers=True,
    )
    # Test that the critical lines and caustics have the correct shape
    assert np.array(critical_lines).ndim == 3
    assert np.array(caustics).ndim == 3
    # Check that the returned centers are consistent with the input kwargs_lens
    assert np.allclose(np.array(centers), np.array([(kw['center_x'], kw['center_y']) for kw in kwargs_model['kwargs_lens'] if 'center_x' in kw]))

def test_shear_deflection_field(lens_image, kwargs_model):
    # Call the tested function
    num_pixels = 15
    x, y, gx, gy = mut.shear_deflection_field(
        lens_image, 
        kwargs_model['kwargs_lens'],
        num_pixels=num_pixels
    )
    # Test that the deflection field has the correct shape
    assert x.shape == (num_pixels, num_pixels)
    assert y.shape == (num_pixels, num_pixels)
    assert gx.shape == (num_pixels, num_pixels)
    assert gy.shape == (num_pixels, num_pixels)

def test_mask_from_source_area(lens_image, kwargs_model):
    mask = mut.mask_from_source_area(lens_image, kwargs_model)
    assert mask.shape == (8, 8)
    # Test that the mask contains only 0s or 1s
    assert np.all(np.isin(mask, [0, 1]))

def test_mask_from_lensed_source(lens_image, kwargs_model):
    mask, binary_source = mut.mask_from_lensed_source(lens_image, kwargs_model)
    assert mask.shape == (8, 8)
    # Test that the mask contains only 0s or 1s
    assert np.all(np.isin(mask, [0, 1]))
    # Same for the binary source
    assert np.all(np.isin(binary_source, [0, 1]))

def test_pixelated_region_from_sersic():
    # Create some fake parameters for a SersicElliptic profile
    kwargs_sersic = {
        'amp': 1.0,
        'R_sersic': 0.5,
        'n_sersic': 4.0,
        'center_x': 0.0,
        'center_y': 0.0,
        'e1': 0.3,
        'e2': -0.1,
    }
    # Call the tested function
    kwargs_pixel_grid = mut.pixelated_region_from_sersic(
        kwargs_sersic, force_square=True,
    )
    # Check that the PixelGrid kwargs represent a square region
    assert kwargs_pixel_grid['grid_shape'][0] == kwargs_pixel_grid['grid_shape'][1]
    # Check that the region is at the same position as the Sersic
    assert np.allclose(
        kwargs_pixel_grid['grid_center'], 
        (kwargs_sersic['center_x'], kwargs_sersic['center_y']),
    )
    # Same when not forcing a square region
    kwargs_pixel_grid = mut.pixelated_region_from_sersic(
        kwargs_sersic, force_square=False,
        min_width=0.1, min_height=0.1,
    )
    assert kwargs_pixel_grid['grid_shape'][0] != kwargs_pixel_grid['grid_shape'][1]
    # Now we test with a non-elliptical Sersic profile (i.e. does not contain e1, e2)
    kwargs_sersic_spherical1 = {
        'amp': 1.0,
        'R_sersic': 0.5,
        'n_sersic': 4.0,
        'center_x': 0.0,
        'center_y': 0.0,
    }
    # We also define other kwargs which contain e1 and e2 but that are zeros
    kwargs_sersic_spherical2 = {
        'amp': 1.0,
        'R_sersic': 0.5,
        'n_sersic': 4.0,
        'center_x': 0.0,
        'center_y': 0.0,
        'e1': 0.0,
        'e2': 0.0,
    }
    # Test that the output of the tested function is the same for both cases
    kwargs_pixel_grid1 = mut.pixelated_region_from_sersic(
        kwargs_sersic_spherical1, force_square=True,
    )
    kwargs_pixel_grid2 = mut.pixelated_region_from_sersic(
        kwargs_sersic_spherical2, force_square=True,
    )
    assert kwargs_pixel_grid1 == kwargs_pixel_grid2

# def test_pixelated_region_from_arc_mask(lens_image, kwargs_model):
#     # Define a fake arc mask image
#     arc_mask = np.zeros((10, 10))
#     arc_mask[2:8, 3:7] = 1
#     # Call the tested function
#     kwargs_pixel_grid = mut.pixelated_region_from_arc_mask(
#         arc_mask, lens_image.Grid, 
#         lens_image.MassModel, kwargs_model['kwargs_lens'],
#     )
#     # Test that the resulting grid settings make senses
#     assert kwargs_pixel_grid['grid_shape'][0] == 6
#     assert kwargs_pixel_grid['grid_shape'][1] == 4

def test_draw_samples_from_covariance():
    # Define a mean and covariance matrix
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    # Call the tested function
    num_samples = 2_000
    samples = mut.draw_samples_from_covariance(mean, cov, num_samples=num_samples, seed=3265)
    # Test that the samples have the correct shape
    assert samples.shape == (num_samples, 2)
    # Test that the samples have the correct covariance
    assert np.allclose(np.cov(samples.T), cov, rtol=1e-1)
