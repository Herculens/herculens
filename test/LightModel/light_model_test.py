# This file provides unit tests using pytest fixtures to extensively test the herculens LightModel class.

import pytest
import numpy as np

from herculens.LightModel.light_model import LightModel
from herculens.LightModel.Profiles.sersic import SersicElliptic
from herculens.LightModel.Profiles.gaussian import GaussianEllipse
from herculens.LightModel.Profiles.shapelets import Shapelets

from herculens.LightModel.profile_mapping import SUPPORTED_MODELS


@pytest.fixture
def base_setup():
    # Some coordinates
    x, y = np.meshgrid(np.linspace(-0.5, 1.5, 5), np.linspace(-1., 1., 5))
    # Create an instance of the LightModel class with some initial parameters
    # Replace the arguments with appropriate values for your use case
    n_max = 4
    light_model = LightModel([
        SersicElliptic(), GaussianEllipse(), Shapelets(n_max=n_max),
    ])

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
            'amps': np.random.randn((n_max+1)*(n_max+2)//2),
            'beta': 0.2,
            'center_x': -0.02,
            'center_y': 0.1,
        }
    ]
    return (x, y), light_model, kwargs_light

def get_light_model_instance(alpha_method):
    # returns a LightModel instance with the different setups that lead
    # to different implementations to compute light profiles
    if alpha_method == 'repeated':
        light_model = LightModel([SersicElliptic(), SersicElliptic(), SersicElliptic()], verbose=True)
    else:
        light_model = LightModel(3 * [SersicElliptic()], verbose=True)
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
    light_model1 = LightModel([SersicElliptic()])
    light_model2 = LightModel(SersicElliptic())
    light_model3 = LightModel('SERSIC_ELLIPSE')  # will be deprecated in the future
    assert isinstance(light_model2.func_list[0], type(light_model1.func_list[0]))
    assert isinstance(light_model3.func_list[0], type(light_model1.func_list[0]))

def test_surface_brightness(base_setup):
    # Test the surface_brightness method
    (x, y), model, kwargs = base_setup
    surface_brightness = model.surface_brightness(x, y, kwargs)
    assert surface_brightness.shape == x.shape

def test_surface_brightness_bis(base_setup):
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
