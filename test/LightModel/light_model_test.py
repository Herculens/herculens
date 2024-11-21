# This file provides unit tests using pytest fixtures to extensively test the herculens LightModel class.

import pytest
import numpy as np

import herculens as hcl
from herculens.LightModel.light_model import LightModel

from herculens.LightModel.profile_mapping import SUPPORTED_MODELS


# Set to True to include the Shapelets profile in the tests, but this requires gigalens
TEST_SHAPELETS = False


@pytest.fixture
def base_setup():
    # Some coordinates
    grid_class = hcl.PixelGrid(nx=5, ny=5)
    x, y = grid_class.pixel_coordinates
    kwargs_pixelated = {'num_pixels': 10}
    # Create an instance of the LightModel class with some initial parameters
    # Replace the arguments with appropriate values for your use case
    profile_list = [
        hcl.SersicElliptic(), 
        hcl.GaussianEllipse(), 
        hcl.PixelatedLight(
            interpolation_type='fast_bilinear', allow_extrapolation=True, 
            derivative_type='interpol', adaptive_grid=False
        )
    ]
    if TEST_SHAPELETS:
        n_max = 4
        profile_list.append(hcl.Shapelets(n_max=n_max))
    light_model = LightModel(
        profile_list, 
        kwargs_pixelated=kwargs_pixelated,
        verbose=True,
    )
    light_model.set_pixel_grid(
        grid_class.create_model_grid(**light_model.pixel_grid_settings),
        data_pixel_area=grid_class.pixel_area,
    )

    # Populate kwargs with parameters associated to the base_light_model
    kwargs_light = [
        {
            'amp': 1.0,
            'R_sersic': 0.5,
            'n_sersic': 4.0,
            'center_x': 0.04,
            'center_y': -0.03,
            'e1': 0.12,
            'e2': 0.07,
        },
        {
            'amp': 0.8,
            'sigma': 0.1,
            'center_x': 0.0,
            'center_y': 0.0,
            'e1': 0.12,
            'e2': 0.034,
        },
        {
            'pixels': np.random.randn(10, 10),
        }
    ]
    if TEST_SHAPELETS:
        kwargs_light.append({
            'amps': np.random.randn((n_max+1)*(n_max+2)//2),
            'beta': 0.2,
            'center_x': -0.02,
            'center_y': 0.1,
        })
    return (x, y), light_model, kwargs_light

def get_light_model_instance(alpha_method):
    # returns a LightModel instance with the different setups that lead
    # to different implementations to compute light profiles
    if alpha_method == 'repeated':
        light_model = LightModel([hcl.SersicElliptic(), hcl.SersicElliptic(), hcl.SersicElliptic()], verbose=True)
    else:
        light_model = LightModel(3 * [hcl.SersicElliptic()], verbose=True)
    kwargs_light = 3 * [
        {
            'amp': 1.0,
            'R_sersic': 0.5,
            'n_sersic': 4.0,
            'center_x': 0.0,
            'center_y': 0.0,
            'e1': 0.0,
            'e2': 0.0,
        }
    ]
    return light_model, kwargs_light

@pytest.mark.parametrize(
    "xy", 
    [(1.0, 1e-6), (1e-6, 1.0), (1.0, 1.0), (1e-6, 1e-6)]
)
def test_summation_methods(xy):
    # unpack the coordinates
    x, y = xy
    # get the instance corresponding to the alpha_method
    light_model1, kwargs_light2 = get_light_model_instance('repeated')
    light_model2, kwargs_light2 = get_light_model_instance('unique')
    # test the resulting values of the light profiles
    assert np.allclose(
        light_model1.surface_brightness(x, y, kwargs_light2), 
        light_model2.surface_brightness(x, y, kwargs_light2), rtol=1e-8
    )
    # here we test the slightly different call when only one profile is evaluated
    assert np.allclose(
        light_model1.surface_brightness(x, y, kwargs_light2, k=0), 
        light_model2.surface_brightness(x, y, kwargs_light2, k=0), rtol=1e-8
    )

def test_single_profile():
    # Create an instance of the LightModel class with a single profile
    light_model1 = LightModel([hcl.SersicElliptic()])
    light_model2 = LightModel(hcl.SersicElliptic())
    light_model3 = LightModel('SERSIC_ELLIPSE')  # will be deprecated in the future
    assert isinstance(light_model2.func_list[0], type(light_model1.func_list[0]))
    assert isinstance(light_model3.func_list[0], type(light_model1.func_list[0]))

def test_surface_brightness(base_setup):
    # Test the surface brightness method
    (x, y), model, kwargs = base_setup
    sb = model.surface_brightness(x, y, kwargs)
    assert sb.shape == x.shape
    assert sb.shape == y.shape
    # Test that the resulting sb corresponds to the sum 
    # of the sb components of individual profiles
    sb_sum = np.zeros_like(x)
    for i, profile in enumerate(model.func_list):
        sb_sum += profile.function(x, y, **kwargs[i])
    assert np.allclose(sb, sb_sum)

def test_spatial_derivatives(base_setup):
    # Test the spatial_derivatives method
    (x, y), model, kwargs = base_setup
    # only the Sersic profile has derivatives implemented so we require k=0
    sb_dx, sb_dy = model.spatial_derivatives(x, y, kwargs, k=0)
    assert sb_dx.shape == x.shape
    assert sb_dy.shape == x.shape
