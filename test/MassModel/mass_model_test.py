# This file provides unit tests using pytest fixtures to extensively test the herculens MassModel class.

import pytest
import numpy as np

from herculens.MassModel.mass_model import MassModel
from herculens.MassModel.Profiles.epl import EPL
from herculens.MassModel.Profiles.shear import ShearGammaPsi
from herculens.MassModel.Profiles.multipole import Multipole

from herculens.MassModel.profile_mapping import SUPPORTED_MODELS


@pytest.fixture
def base_mass_model():
    # Create an instance of the MassModel class with some initial parameters
    # Replace the arguments with appropriate values for your use case
    return MassModel([
        EPL(), ShearGammaPsi(), Multipole(),
    ], use_jax_scan=False)

@pytest.fixture
def base_kwargs_mass():
    # Populate kwargs with parameters associated to the base_mass_model
    return [
        {
            'theta_E': 1.0,
            'gamma': 2.0,
            'center_x': 0.0,
            'center_y': 0.0,
            'e1': 0.0,
            'e2': 0.0,
        },
        {
            'gamma_ext': 0.1,
            'psi_ext': 0.0,
        },
        {
            'm': 4.,
            'a_m': 0.1,
            'phi_m': 0.,
            'center_x': 0.,
            'center_y': 0.,
        }
    ]

def get_mass_model_instance(alpha_method):
    # returns a MassModel instance with the different setups that lead
    # to different methods to compute deflection angles
    if alpha_method in ('unique', 'repeated'):
        if alpha_method == 'repeated':
            mass_model = MassModel([EPL(), EPL(), EPL()], verbose=True)
        else:
            mass_model = MassModel(3 * [EPL()], verbose=True)
        kwargs_mass = 3 * [
            {
                'theta_E': 1.0,
                'gamma': 2.0,
                'center_x': 0.0,
                'center_y': 0.0,
                'e1': 0.0,
                'e2': 0.0,
            }
        ]
    elif alpha_method in ('scan', 'loop'):
        if alpha_method == 'scan':
            mass_model = MassModel([
                EPL(), ShearGammaPsi(), Multipole(),
            ], use_jax_scan=True)
        else:
            mass_model = MassModel([
                EPL(), ShearGammaPsi(), Multipole(),
            ], use_jax_scan=False)
        kwargs_mass = [
            {
                'theta_E': 1.0,
                'gamma': 2.0,
                'center_x': 0.0,
                'center_y': 0.0,
                'e1': 0.0,
                'e2': 0.0,
            },
            {
                'gamma_ext': 0.1,
                'psi_ext': 0.0,
            },
            {
                'm': 4.,
                'a_m': 0.1,
                'phi_m': 0.,
                'center_x': 0.,
                'center_y': 0.,
            }
        ]
    return mass_model, kwargs_mass

@pytest.mark.parametrize(
    "comparisons", 
    [
        ('repeated', 'unique'), 
        ('scan', 'loop'),
    ]
)
@pytest.mark.parametrize(
    "xy", 
    [(1.0, 1e-6), (1e-6, 1.0), (1.0, 1.0), (1e-6, 1e-6)]
)
def test_alpha_methods(xy, comparisons):
    # unpack the coordinates
    x, y = xy
    # unpack the comparisons
    alpha_method1, alpha_method2 = comparisons
    print("WTF", alpha_method1, alpha_method2)
    # get the instance corresponding to the alpha_method
    mass_model1, kwargs_mass2 = get_mass_model_instance(alpha_method1)
    mass_model2, kwargs_mass2 = get_mass_model_instance(alpha_method2)
    # test the resulting values of the deflection angles
    assert np.allclose(
        mass_model1.alpha(x, y, kwargs_mass2), 
        mass_model2.alpha(x, y, kwargs_mass2), rtol=1e-8
    )
    # here we test the slightly different call when only one profile is evaluated
    assert np.allclose(
        mass_model1.alpha(x, y, kwargs_mass2, k=0), 
        mass_model2.alpha(x, y, kwargs_mass2, k=0), rtol=1e-8
    )

def test_single_profile():
    # Create an instance of the MassModel class with a single profile
    mass_model1 = MassModel([EPL()])
    mass_model2 = MassModel(EPL())
    mass_model3 = MassModel('EPL')  # will be deprecated in the future
    assert isinstance(mass_model2.func_list[0], type(mass_model1.func_list[0]))
    assert isinstance(mass_model3.func_list[0], type(mass_model1.func_list[0]))
