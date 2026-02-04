# This file provides unit tests using pytest fixtures to extensively test the herculens MassModel class.

import pytest
import numpy as np

from herculens.MassModel.mass_model import MassModel
from herculens.MassModel.Profiles.epl import EPL
from herculens.MassModel.Profiles.shear import ShearGammaPhi
from herculens.MassModel.Profiles.multipole import Multipole

from herculens.MassModel.profile_mapping import SUPPORTED_MODELS


@pytest.fixture
def base_setup():
    # Some coordinates
    x, y = np.meshgrid(np.linspace(-0.5, 1.5, 5), np.linspace(-1., 1., 5))
    # Create an instance of the MassModel class with some initial parameters
    # Replace the arguments with appropriate values for your use case
    mass_model = MassModel([
        EPL(), ShearGammaPhi(), Multipole(),
    ], use_jax_scan=False)

    # Populate kwargs with parameters associated to the base_mass_model
    kwargs_mass = [
        {
            'theta_E': 1.5,
            'gamma': 2.1,
            'center_x': 0.04,
            'center_y': -0.03,
            'e1': 0.12,
            'e2': 0.07,
        },
        {
            'gamma_ext': 0.1,
            'psi_ext': 0.8,
        },
        {
            'm': 4.,
            'a_m': 0.1,
            'phi_m': 0.,
            'center_x': -0.02,
            'center_y': 0.1,
        }
    ]
    return (x, y), mass_model, kwargs_mass

def get_mass_model_instance(alpha_method):
    # returns a MassModel instance with the different setups that lead
    # to different implementations to compute deflection angles
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
                EPL(), ShearGammaPhi(), Multipole(),
            ], use_jax_scan=True)
        else:
            mass_model = MassModel([
                EPL(), ShearGammaPhi(), Multipole(),
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
def test_summation_methods(xy, comparisons):
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

def test_ray_shooting(base_setup):
    # Test the ray_shooting method
    (x, y), model, kwargs = base_setup
    x_image, y_image = model.ray_shooting(x, y, kwargs)
    assert x_image.shape == x.shape
    assert y_image.shape == y.shape

def test_potential(base_setup):
    # Test the potential method
    (x, y), model, kwargs = base_setup
    potential = model.potential(x, y, kwargs)
    assert potential.shape == x.shape

def test_fermat_potential(base_setup):
    # Test the fermat_potential method
    (x, y), model, kwargs = base_setup
    fermat_potential = model.fermat_potential(x, y, kwargs)
    assert fermat_potential.shape == x.shape

def test_kappa(base_setup):
    # Test the kappa method
    (x, y), model, kwargs = base_setup
    kappa = model.kappa(x, y, kwargs)
    assert kappa.shape == x.shape

def test_curl(base_setup):
    # Test the curl method
    (x, y), model, kwargs = base_setup
    curl = model.curl(x, y, kwargs)
    assert curl.shape == x.shape

def test_alpha(base_setup):
    # Test the alpha method
    (x, y), model, kwargs = base_setup
    alpha_x, alpha_y = model.alpha(x, y, kwargs)
    assert alpha_x.shape == x.shape
    assert alpha_y.shape == y.shape

def test_hessian(base_setup):
    # Test the hessian method
    (x, y), model, kwargs = base_setup
    hessian = model.hessian(x, y, kwargs)
    hessian_stacked = np.stack(hessian)
    assert hessian_stacked.shape == (4, x.shape[0], x.shape[1])
    # Test that the second component of the hessian is equal to the third component
    assert np.allclose(hessian_stacked[1], hessian_stacked[2])
    # Test that the resulting hessian corresponds to the sum 
    # of the hessian components of individual profiles
    hessian_sum = np.zeros((3, x.shape[0], x.shape[1]))
    for i, profile in enumerate(model.func_list):
        hess_i = np.array(profile.hessian(x, y, **kwargs[i]))
        if hess_i.ndim == 1: hess_i = hess_i[:, np.newaxis, np.newaxis]
        hessian_sum += hess_i
    # For comparison we take into account that the profile hessian 
    # does not return the same components and in the same order.
    # We also do not compare the third element (equal to second by definition)
    assert np.allclose(hessian_stacked[[0, 3, 1], :, :], hessian_sum)

def test_gamma(base_setup):
    # Test the gamma method
    (x, y), model, kwargs = base_setup
    gamma_x, gamma_y = model.gamma(x, y, kwargs)
    assert gamma_x.shape == x.shape
    assert gamma_y.shape == y.shape

def test_magnification(base_setup):
    # Test the magnification method
    (x, y), model, kwargs = base_setup
    magnification = model.magnification(x, y, kwargs)
    assert magnification.shape == x.shape
    magnification = model.magnification(x, y, kwargs, k=0)
    assert magnification.shape == x.shape
