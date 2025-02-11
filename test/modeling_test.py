# Testing modeling workflows
# 
# Copyright (c) 2023, herculens developers and contributors


import numpy as np
import numpy.testing as npt
import pytest
from copy import deepcopy

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.LensImage.lens_image import LensImage
from herculens.Inference.loss import Loss
from herculens.Inference.ProbModel.numpyro import NumpyroModel
from herculens.Inference.Optimization.jaxopt import JaxoptOptimizer


def simulate_data(data_type, supersampling_factor):
    npix = 40  # number of pixel on a side
    pix_scl = 0.16  # pixel size in arcsec
    half_size = npix * pix_scl / 2.
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2.
    transform_pix2angle = pix_scl * np.eye(2)
    pixel_grid = PixelGrid(nx=npix, ny=npix, 
                           ra_at_xy_0=ra_at_xy_0, dec_at_xy_0=dec_at_xy_0,
                           transform_pix2angle=transform_pix2angle)
    psf = PSF(psf_type='GAUSSIAN', fwhm=0.3, pixel_size=pix_scl)
    noise = Noise(npix, npix, background_rms=1e-2, exposure_time=2000.)

    # Define simulation parameters
    kwargs_lens_mass_input = [
        {'theta_E': 1.5, 'gamma': 2.1, 'e1': 0.1, 'e2': -0.05, 'center_x': 0., 'center_y': 0.},  # power-law
        {'gamma1': -0.03, 'gamma2': 0.02, 'ra_0': 0., 'dec_0': 0.}  # external shear
    ]

    # Lens light
    kwargs_lens_light_input = [
        {'amp': 8.0, 'R_sersic': 1.4, 'n_sersic': 3., 'e1': 0.1, 'e2': -0.05, 'center_x': 0., 'center_y': 0.}  # elliptical Sérsic
    ]

    # Source light
    kwargs_source_input = [
        {'amp': 5.0, 'R_sersic': 0.2, 'n_sersic': 2., 'e1': -0.05, 'e2': 0.05, 'center_x': 0.05, 'center_y': 0.1}
    ]

    # Define input model components
    if data_type == 'full':
        lens_mass_input = MassModel(['EPL', 'SHEAR'])
        lens_light_input = LightModel(['SERSIC_ELLIPSE'])
        source_input = LightModel(['SERSIC_ELLIPSE'])
    elif data_type == 'lens_light_only':
        lens_mass_input = MassModel([])
        lens_light_input = LightModel(['SERSIC_ELLIPSE'])
        source_input = LightModel([])
    elif data_type == 'lensed_source_only':
        lens_mass_input = MassModel(['EPL', 'SHEAR'])
        lens_light_input = LightModel([])
        source_input = LightModel(['SERSIC_ELLIPSE'], verbose=True)
    elif data_type == 'source_only':
        lens_mass_input = MassModel([])
        lens_light_input = LightModel([])
        source_input = LightModel(['SERSIC_ELLIPSE'])

    lens_image_input = LensImage(pixel_grid, psf, noise_class=noise,
                                 lens_mass_model_class=lens_mass_input,
                                 source_model_class=source_input,
                                 lens_light_model_class=lens_light_input,
                                 kwargs_numerics={'supersampling_factor': supersampling_factor})

    kwargs_input = dict(kwargs_lens=kwargs_lens_mass_input,
                            kwargs_source=kwargs_source_input,
                            kwargs_lens_light=kwargs_lens_light_input)
        
    data = lens_image_input.simulation(
        **kwargs_input, compute_true_noise_map=True, 
        add_poisson_noise=True, add_background_noise=True,
        prng_key=jax.random.PRNGKey(0),
    )
    return data, lens_image_input, kwargs_input


@pytest.mark.parametrize(
    "data_type",
    [
        'full', 
        'lensed_source_only', 
        'source_only', 
        'lens_light_only'
    ]
)
@pytest.mark.parametrize(
    "supersampling_factor",
    [
        1, 
        2, 
        3,
    ],
)
def test_model_fit(data_type, supersampling_factor):
    # Get some fake imaging data
    data, lens_image_input, kwargs_input = simulate_data(data_type, supersampling_factor)

    # Define model components to fit
    lens_image_fit = LensImage(
        deepcopy(lens_image_input.Grid), 
        deepcopy(lens_image_input.PSF), 
        noise_class=deepcopy(lens_image_input.Noise),
        lens_mass_model_class=deepcopy(lens_image_input.MassModel),
        source_model_class=deepcopy(lens_image_input.SourceModel),
        lens_light_model_class=deepcopy(lens_image_input.LensLightModel),
        kwargs_numerics={'supersampling_factor': supersampling_factor}
    )

    # Define the probabilistic model
    class ProbModel(NumpyroModel):
    
        def model(self):
            # Parameters of the source
            prior_source = [
              {
                  'amp': numpyro.sample('source_amp', dist.LogNormal(np.log10(8.), 0.1)),
             'R_sersic': numpyro.sample('source_R_sersic', dist.TruncatedNormal(0.1, 0.05, low=0.02)), 
             'n_sersic': numpyro.sample('source_n', dist.Uniform(1., 3.)), 
             'e1': numpyro.sample('source_e1', dist.TruncatedNormal(0.1, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('source_e2', dist.TruncatedNormal(-0.05, 0.05, low=-0.3, high=0.3)),
             'center_x': numpyro.sample('source_center_x', dist.TruncatedNormal(0., 0.05, low=-0.2, high=0.2)), 
            'center_y': numpyro.sample('source_center_y', dist.TruncatedNormal(0., 0.05, low=-0.2, high=0.2))}
            ]

            # Parameters of the lens
            cx = numpyro.sample('lens_center_x', dist.TruncatedNormal(0., 0.05, low=-0.1, high=0.1))
            cy = numpyro.sample('lens_center_y', dist.TruncatedNormal(0., 0.05, low=-0.1, high=0.1))
            prior_lens = [
                # power-law
            {
                'theta_E': numpyro.sample('lens_theta_E', dist.Normal(1.5, 0.1)),
                'gamma': numpyro.sample('lens_gamma', dist.Normal(2.1, 0.05)),
             'e1': numpyro.sample('lens_e1', dist.TruncatedNormal(0.1, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('lens_e2', dist.TruncatedNormal(-0.05, 0.05, low=-0.3, high=0.3)),
             'center_x': cx, 
             'center_y': cy},
                # external shear, with fixed origin
            {'gamma1': numpyro.sample('lens_gamma1', dist.TruncatedNormal(-0.03, 0.05, low=-0.3, high=0.3)), 
             'gamma2': numpyro.sample('lens_gamma2', dist.TruncatedNormal(0.02, 0.05, low=-0.3, high=0.3)), 
             'ra_0': 0.0, 'dec_0': 0.0}
            ]

            # Parameters of the lens light, with center relative the lens mass
            prior_lens_light = [
            {'amp': numpyro.sample('light_amp', dist.LogNormal(np.log10(5.), 0.1)) , 
             'R_sersic': numpyro.sample('light_R_sersic', dist.Normal(1.3, 0.05)), 
             'n_sersic': numpyro.sample('light_n', dist.Uniform(2., 4.)), 
             'e1': numpyro.sample('light_e1', dist.TruncatedNormal(-0.05, 0.05, low=-0.3, high=0.3)),
             'e2': numpyro.sample('light_e2', dist.TruncatedNormal(0.05, 0.05, low=-0.3, high=0.3)),
             'center_x': numpyro.sample('light_center_x', dist.Normal(cx, 0.01)), 
             'center_y': numpyro.sample('light_center_y', dist.Normal(cy, 0.01))}
            ]
            
            # wrap up all parameters for the lens_image.model() method
            if data_type == 'full':
                model_params = dict(kwargs_lens=prior_lens, 
                                    kwargs_lens_light=prior_lens_light,
                                    kwargs_source=prior_source)
            elif data_type == 'lensed_source_only':
                model_params = dict(kwargs_lens=prior_lens, 
                                    kwargs_source=prior_source)
            elif data_type == 'source_only':
                model_params = dict(kwargs_source=prior_source)
            elif data_type == 'lens_light_only':
                model_params = dict(kwargs_lens_light=prior_lens_light)
            
            # generates the model image
            model_image = lens_image_fit.model(**model_params)
            
            # estimate the error per pixel
            model_error = jnp.sqrt(lens_image_fit.Noise.C_D_model(model_image))
            
            # finally defines the observed node, conditioned on the data assuming a Gaussian distribution
            numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_error), 2), obs=data)
        
        def params2kwargs(self, params):
            if data_type in ['lensed_source_only', 'full']:
                kwargs_lens = [{'theta_E': params['lens_theta_E'],
                'gamma': params['lens_gamma'],
                'e1': params['lens_e1'],
                'e2': params['lens_e2'],
                'center_x': params['lens_center_x'],
                'center_y': params['lens_center_y']},
                {'gamma1': params['lens_gamma1'],
                'gamma2': params['lens_gamma2'],
                'ra_0': 0.,
                'dec_0': 0.}]
            else:
                kwargs_lens = None
            if data_type in ['lensed_source_only', 'source_only', 'full']:
                kwargs_source = [{'amp': params['source_amp'],
                'R_sersic': params['source_R_sersic'],
                'n_sersic': params['source_n'],
                'e1': params['source_e1'],
                'e2': params['source_e2'],
                'center_x': params['source_center_x'],
                'center_y': params['source_center_y']}]
            else:
                kwargs_source = None
            if data_type in ['lens_light_only', 'full']:
                kwargs_lens_light = [{'amp': params['light_amp'],
                'R_sersic': params['light_R_sersic'],
                'n_sersic': params['light_n'],
                'e1': params['light_e1'],
                'e2': params['light_e2'],
                'center_x': params['light_center_x'],
                'center_y': params['light_center_y']}]
            else:
                kwargs_lens_light = None
            return dict(kwargs_lens=kwargs_lens,
                        kwargs_source=kwargs_source,
                        kwargs_lens_light=kwargs_lens_light)

    prob_model = ProbModel()
    n_param = prob_model.num_parameters
    # print("Number of parameters:", n_param)

    # Defines the loss function
    loss = Loss(prob_model)

    # Draw some initial parameter values (from the prior)
    init_params = prob_model.unconstrain(prob_model.get_sample(prng_key=jax.random.PRNGKey(1)))
    kwargs_init = prob_model.params2kwargs(prob_model.constrain(init_params))
    print("Initial loss =", loss(init_params))
    print("Initial gradient =", loss.gradient(init_params))

    model_init = lens_image_fit.model(**kwargs_init)
    red_chi2_init = lens_image_fit.reduced_chi2(data, model_init)
    assert red_chi2_init > 1.05  # residual should be bad here

    # uncomment this to check the initial model compared to the data
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(data)
    # axes[1].imshow(model_init)
    # plt.show()
    # # raise

    # Performs the fit
    optimizer = JaxoptOptimizer(loss, loss_norm_optim=data.size)
    bestfit_params, logL_best_fit, extra_fields, runtime \
        = optimizer.run_scipy(init_params, method='BFGS', maxiter=200)

    # uncomment this to check the evolution of the loss
    # plt.figure()
    # plt.plot(extra_fields['loss_history'])
    # plt.show()

    kwargs_bestfit = prob_model.params2kwargs(prob_model.constrain(bestfit_params))

    model_bestfit = lens_image_fit.model(**kwargs_bestfit)
    residuals, _ = lens_image_fit.normalized_residuals(data, model_bestfit)
    red_chi2_bestfit = lens_image_fit.reduced_chi2(data, model_bestfit)

    # plt.figure()
    # plt.imshow(residuals, vmin=-3, vmax=3)
    # plt.show()

    # assert that residuals after fitting are effectively down to the noise
    assert red_chi2_bestfit < 1.05
