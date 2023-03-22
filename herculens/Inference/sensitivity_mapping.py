# Gradient-based sensitivity mapping
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import copy
import time
import numpy as np
from skimage import feature
from functools import partial

from jax import grad
from jax import jit
import jax.numpy as jnp

from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.LensImage.lens_image import LensImage
from herculens.Inference.legacy.parameters import Parameters
from herculens.Inference.legacy.loss import Loss
from herculens.Inference.legacy.optimization import Optimizer
# from herculens.Inference.legacy.covariance import FisherCovariance
from herculens.Util import util


__all__ = ['MassSensitivityMapping']



class MassSensitivityMapping(object):

    _MODELS_SUPPORTED = [
        'POINT_MASS',       # point mass (analytical)
        'SIS',              # Singuler Isothermal Sphere (analytical)
        'PIXELATED_DIRAC',  # single potential 'pixel'
        # TODO: add NFW and pseudo-Jaffe profiles
    ]

    def __init__(self, data, macro_lens_image, macro_parameters, macro_loss, 
                 halo_profile='SIS', fix_macro_parameters=True, 
                 kwargs_numerics=None, verbose=False):
        self.data = data
        self.m_lens_image = macro_lens_image
        self.m_param = macro_parameters
        self.m_loss = macro_loss
        self.kwargs_numerics = kwargs_numerics
        self.fix_macro = fix_macro_parameters
        self.halo_profile = halo_profile
        self.verbose = verbose

    def sensitivity_map(self, init_mass=0., x_grid=None, y_grid=None, 
                        use_jax_vectorize=False):
        # prepare the new model
        self.prepare_halo_model(init_mass)

        # create the loss corresponding this new model
        self.halo_loss = Loss(self.data, self.halo_lens_image, self.halo_param, 
                              likelihood_type='chi2')

        # define the grid on which to compute sensitivity
        if self.halo_lens_image.ImageNumerics.grid_supersampling_factor > 1:
            x_grid, y_grid = self.halo_lens_image.ImageNumerics.coordinates_evaluate # 1D arrays
        elif x_grid is None or y_grid is None:
            x_grid, y_grid = self.halo_lens_image.Grid.pixel_coordinates

        # CHECKS
        if self.verbose:
            print("halo loss at edge:", self.halo_loss([init_mass, x_grid[0, 0], y_grid[0, 0]]))
            print("halo grad loss at edge:", self.halo_loss.gradient([init_mass, x_grid[0, 0], y_grid[0, 0]]))

            print("macro loss:", self.m_loss(self.p_macro))
            # print("macro grad loss:", grad(self.m_loss)(self.p_macro))

        @jit
        def sensitivity_at_pixel(x, y):
            if self.fix_macro:
                p = [init_mass, x, y]
            else:
                p = [init_mass, x, y] + self.p_macro
            grad_loss_mass = self.halo_loss.gradient(p)
            partial_deriv_mass = grad_loss_mass[0]
            return partial_deriv_mass
        
        # evaluate the sensitivity over the coordinates grid
        start = time.time()
        if use_jax_vectorize:
            sensitivity_map = jnp.vectorize(sensitivity_at_pixel)(x_grid, y_grid).block_until_ready()
        else:
            sensitivity_map = np.vectorize(sensitivity_at_pixel)(x_grid, y_grid)
        runtime = time.time() - start

        # convert to numpy array and reshape
        sensitivity_map = np.array(sensitivity_map)
        if len(sensitivity_map.shape) == 1:
            sensitivity_map = util.array2image(sensitivity_map)

        # get the coordinates where the sensitivity is the highest
        peak_indices_2d = feature.peak_local_max(-np.clip(sensitivity_map, a_min=None, a_max=0))
        
        x_coords, y_coords = x_grid[0, :], y_grid[:, 0]
        x_minima = x_coords[peak_indices_2d[:, 1]]
        y_minima = y_coords[peak_indices_2d[:, 0]]
        z_minima = sensitivity_map[peak_indices_2d[:, 0], peak_indices_2d[:, 1]]

        # if with_covariance:
        #     @jit
        #     def covariance_at_pixel(x, y):
        #         if self.fix_macro:
        #             p = [init_mass, x, y]
        #         else:
        #             p = [init_mass, x, y] + self.p_macro
        #         grad_loss_mass = self.halo_loss.hessian(p)
        #         partial_deriv_mass = grad_loss_mass[0]
        #         return partial_deriv_mass
        #     covariance_cube = np.vectorize(covariance_at_pixel)(x_grid, y_grid)

        return sensitivity_map, (x_minima, y_minima, z_minima), runtime

    def sensitivity_map_optim(self, init_mass=0., x_grid=None, y_grid=None, minimize_method='trust-krylov', **kwargs_optimizer):
        # prepare the new model
        self.prepare_halo_model(init_mass)

        # create the loss corresponding this new model
        self.halo_loss = Loss(self.data, self.halo_lens_image, self.halo_param, likelihood_type='chi2')

        # create the optimizer
        optimizer = Optimizer(self.halo_loss, self.halo_param)

        # define the grid on which to compute sensitivity
        if self.halo_lens_image.ImageNumerics.grid_supersampling_factor > 1:
            x_grid, y_grid = self.halo_lens_image.ImageNumerics.coordinates_evaluate # 1D arrays
        elif x_grid is None or y_grid is None:
            x_grid, y_grid = self.halo_lens_image.Grid.pixel_coordinates

        # CHECKS
        if self.verbose:
            print("halo loss at edge:", self.halo_loss([init_mass, x_grid[0, 0], y_grid[0, 0]]))
            print("halo grad at edge:", self.halo_loss.gradient([init_mass, x_grid[0, 0], y_grid[0, 0]]))

            print("macro loss:", self.m_loss(self.p_macro))
            # print("macro grad loss:", grad(self.m_loss)(self.p_macro))

        def sensitivity_at_pixel(x, y):
            if self.fix_macro:
                p = [init_mass, x, y]
            else:
                p = [init_mass, x, y] + self.p_macro
            best_fit, logL, _, runtime = optimizer.minimize(method=minimize_method, init_params=p, **kwargs_optimizer)
            return float(best_fit[0]), logL, runtime
        
        # evaluate the sensitivity over the coordinates grid
        start = time.time()
        sensitivity_map, logL_map, runtimes = np.vectorize(sensitivity_at_pixel)(x_grid, y_grid)
        runtime = np.sum(runtimes)

        # convert to numpy array and reshape
        sensitivity_map = np.array(sensitivity_map)
        logL_map = np.array(logL_map)
        if len(sensitivity_map.shape) == 1:
            sensitivity_map = util.array2image(sensitivity_map)
            logL_map = util.array2image(logL_map)

        # get the coordinates where the sensitivity is the highest
        peak_indices_2d = feature.peak_local_max(-np.clip(sensitivity_map, a_min=None, a_max=0))
        
        x_coords, y_coords = x_grid[0, :], y_grid[:, 0]
        x_minima = x_coords[peak_indices_2d[:, 1]]
        y_minima = y_coords[peak_indices_2d[:, 0]]
        z_minima = sensitivity_map[peak_indices_2d[:, 0], peak_indices_2d[:, 1]]

        # if with_covariance:
        #     @jit
        #     def covariance_at_pixel(x, y):
        #         if self.fix_macro:
        #             p = [init_mass, x, y]
        #         else:
        #             p = [init_mass, x, y] + self.p_macro
        #         grad_loss_mass = self.halo_loss.hessian(p)
        #         partial_deriv_mass = grad_loss_mass[0]
        #         return partial_deriv_mass
        #     covariance_cube = np.vectorize(covariance_at_pixel)(x_grid, y_grid)

        return sensitivity_map, (x_minima, y_minima, z_minima), logL_map, runtime

    def prepare_halo_model(self, init_mass):
        # value at which the gradient will be evaluated
        init_mass = init_mass

        if self.halo_profile == 'POINT_MASS':
            kwargs_halo_fixed = {}
            kwargs_halo_init = {'theta_E': init_mass, 'center_x': 0., 'center_y': 0.}
        elif self.halo_profile == 'PIXELATED_DIRAC':
            kwargs_halo_fixed = {}
            kwargs_halo_init = {'psi': init_mass, 'center_x': 0., 'center_y': 0.}
        elif self.halo_profile == 'SIS':
            kwargs_halo_fixed = {}
            kwargs_halo_init = {'theta_E': init_mass, 'center_x': 0., 'center_y': 0.}
        else:
            raise NotImplementedError(f"Halo profile must be in {self._MODELS_SUPPORTED}.")
        
        halo_mass_model_list = [self.halo_profile] + self.m_lens_image.MassModel.profile_type_list
        halo_mass_model = MassModel(halo_mass_model_list)

        grid = copy.deepcopy(self.m_lens_image.Grid)
        #grid.remove_model_grid('lens')
        psf = copy.deepcopy(self.m_lens_image.PSF)
        noise = copy.deepcopy(self.m_lens_image.Noise)
        self.halo_lens_image = LensImage(grid, psf, noise_class=noise,
                                    lens_mass_model_class=halo_mass_model,
                                    source_model_class=self.m_lens_image.SourceModel,
                                    lens_light_model_class=self.m_lens_image.LensLightModel,
                                    kwargs_numerics=self.kwargs_numerics)

        kwargs_macro = self.m_param.current_values(as_kwargs=True)
        self.p_macro = copy.deepcopy(self.m_param.current_values(as_kwargs=False)).tolist()

        if self.fix_macro:
            kwargs_fixed = {
                'kwargs_lens': [kwargs_halo_fixed] + kwargs_macro['kwargs_lens'],
                'kwargs_source': kwargs_macro['kwargs_source'],
                'kwargs_lens_light': kwargs_macro['kwargs_lens_light'],
            }
        else:
            kwargs_fixed = {
                'kwargs_lens': [kwargs_halo_fixed] + [{} for _ in range(len(kwargs_macro['kwargs_lens']))],
                'kwargs_source': [{} for _ in range(len(kwargs_macro['kwargs_source']))],
                'kwargs_lens_light': kwargs_macro['kwargs_lens_light'],
            }
            # some parameters still need to be fixed
            for i, kwargs_profile in enumerate(kwargs_macro['kwargs_lens']):
                for key, value in kwargs_profile.items():
                    if key in ['ra_0', 'dec_0']:  # external shear origin
                        kwargs_fixed['kwargs_lens'][i+1][key] = value
                    elif key in ['center_x', 'center_y']:  # macro lens center (ATTENTION if this is not wanted)
                        kwargs_fixed['kwargs_lens'][i+1][key] = value

        kwargs_init = {
            'kwargs_lens': [kwargs_halo_init] + kwargs_macro['kwargs_lens'],
            'kwargs_source': kwargs_macro['kwargs_source'],
            'kwargs_lens_light': kwargs_macro['kwargs_lens_light'],
        }
        self.halo_param = Parameters(self.halo_lens_image, kwargs_init, kwargs_fixed)

        if self.verbose:
            print("parameters:", self.halo_param.names)
            print("num. params:", self.halo_param.num_parameters)
            print("init. values:", self.halo_param.initial_values())
