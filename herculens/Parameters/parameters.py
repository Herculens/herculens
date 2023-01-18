# Estimation of model parameters storage and manipulation
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


from copy import deepcopy
import numpy as np
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

from herculens.MassModel.Profiles import pixelated as pixelated_lens
from herculens.MassModel.Profiles import (epl, sie, sis, nie, shear, point_mass, 
                                          gaussian_potential, multipole)
from herculens.LightModel.Profiles import pixelated as pixelated_light
from herculens.LightModel.Profiles import gaussian, sersic, uniform, shapelets
from herculens.MassModel.mass_model_base import SUPPORTED_MODELS as MASS_MODELS
from herculens.LightModel.light_model_base import SUPPORTED_MODELS as LIGHT_MODELS
from herculens.Util import model_util


__all__ = ['Parameters']


class Parameters(object):
    """Class that manages parameters in JAX / auto-differentiable framework.
    Currently, it handles:
    - conversions from the differentiable parameter vector to user-friendly dictionnaries (args-to-kwargs, kwargs-to-args)
    - uniform and gaussian priors for subsets of parameters
    - log-prior values, that are meant to be added to the full loss function
    - nice LaTeX format for parameter names

    Example syntax for kwargs_joint:
    'lens_with_lens_light' : [
        [
            (lens_light_index1, lens_index1), [(lens_light_param1, lens_param1), (lens_light_param2, lens_param2), ...]
        ],
        [
            (lens_light_index2, lens_index2), [(lens_light_param3, lens_param3), (lens_light_param4, lens_param4), ...]
        ],
        ...
    ]

    Alternatively, if the parameters share the same name:
    'lens_with_lens_light' : [
        [
            (lens_light_index1, lens_index1), [param1, param2, ...]
        ],
        [
            (lens_light_index2, lens_index2), [param3, param4, ...]
        ],
        ...
    ]
    """

    _unif_prior_penalty = 1e10

    def __init__(self, lens_image, kwargs_init, kwargs_fixed, 
                 kwargs_prior=None, kwargs_joint=None):
        self._image = lens_image
        self._kwargs_init  = kwargs_init
        self._kwargs_fixed = kwargs_fixed
        num_lens_profiles = max(1, len(self._image.MassModel.profile_type_list))
        num_source_profiles = max(1, len(self._image.SourceModel.profile_type_list))
        num_lens_light_profiles = max(1, len(self._image.LensLightModel.profile_type_list))
        kwargs_prior_tmp = {
            'kwargs_lens': [{} for _ in range(num_lens_profiles)],
            'kwargs_source': [{} for _ in range(num_source_profiles)],
            'kwargs_lens_light': [{} for _ in range(num_lens_light_profiles)],
        }
        if kwargs_prior is not None:
            kwargs_prior_tmp.update(kwargs_prior)
        self._kwargs_prior = kwargs_prior_tmp
        kwargs_joint_tmp = {
            'lens_with_lens': [],
            'source_with_source': [],
            'lens_light_with_lens_light': [],
            'lens_with_lens_light': [],
        }
        if kwargs_joint is not None:
            kwargs_joint_tmp.update(kwargs_joint)
        self._kwargs_joint = kwargs_joint_tmp
        self._update_arrays()

        # TODO: write function that checks that no fields are missing
        # and fill those with default values if needed

    @property
    def optimized(self):
        return hasattr(self, '_kwargs_map')

    @property
    def num_parameters(self):
        return self._num_params

    @property
    def prior_types(self):
        return self._prior_types

    @property
    def bounds(self):
        return self._lowers, self._uppers

    def draw_prior_samples(self, num_samples, seed=None):
        """for a parameter that has no prior, returns its current value"""
        if seed is not None:
            np.random.seed(seed)
        samples = []
        for n in range(num_samples):
            param_values = []
            for i in range(self.num_parameters):
                if self._prior_types[i] == 'gaussian':
                    param_value = self._means[i] + self._widths[i] * np.random.randn()
                elif self._prior_types[i] == 'uniform':
                    param_value = np.random.uniform(low=self._lowers[i], high=self._uppers[i])
                else:
                    param_value = self.current_values()[i]
                param_values.append(param_value)
            samples.append(param_values)
        return np.array(samples)

    @property
    def names(self):
        if not hasattr(self, '_names'):
            self._names = self._set_names('mass_model_list', 'kwargs_lens')
            self._names += self._set_names('source_model_list', 'kwargs_source')
            self._names += self._set_names('lens_light_model_list', 'kwargs_lens_light')
        return self._names

    @property
    def symbols(self):
        if not hasattr(self, '_symbols'):
            self._symbols = self._name2latex(self.names)
        return self._symbols

    @property
    def kwargs_model(self):
        # TODO: intermediate step, this might be suppressed in the future
        self._kwargs_model = dict(mass_model_list=self._image.MassModel.profile_type_list,
                                  source_model_list=self._image.SourceModel.profile_type_list,
                                  lens_light_model_list=self._image.LensLightModel.profile_type_list)
        return self._kwargs_model

    def initial_values(self, as_kwargs=False, copy=False):
        if as_kwargs:
            return deepcopy(self._kwargs_init) if copy else self._kwargs_init
        else:
            return deepcopy(self._init_values) if copy else self._init_values

    def current_values(self, as_kwargs=False, restart=False, copy=False):
        if restart is True or not self.optimized:
            return self.initial_values(as_kwargs=as_kwargs, copy=copy)
        return self.best_fit_values(as_kwargs=as_kwargs, copy=copy)

    def best_fit_values(self, as_kwargs=False, copy=False):
        """Maximum-a-postriori estimate"""
        if as_kwargs:
            return deepcopy(self._kwargs_map) if copy else self._kwargs_map
        else:
            return deepcopy(self._map_values) if copy else self._map_values

    def samples(self, as_kwargs=False, group_by_param=False):
        if not hasattr(self, '_samples'):
            return None
        if as_kwargs:
            # TODO: save in cache + create separate class ParametersSamples to handle this in a cleaner way
            num_samples = self._samples.shape[0]
            kwargs_samples = [self.args2kwargs(self._samples[i, :]) for i in range(num_samples)]
            if group_by_param:
                kwargs_samples = self._group_kwargs_samples_by_param(kwargs_samples)
            return kwargs_samples
        return self._samples

    @property
    def covariance_matrix(self):
        if hasattr(self, '_cov_matrix'):
            return self._cov_matrix
        elif hasattr(self, '_samples'):
            raise NotImplementedError("Estimate covariance from samples is not yet supported.")
        else:
            return None

    def set_best_fit(self, args):
        self._map_values = args
        self._kwargs_map = self.args2kwargs(self._map_values)
    
    def set_posterior_samples(self, samples, losses):
        min_loss_idx = np.argmin(losses)
        self._map_values = samples[min_loss_idx, :]
        self._kwargs_map = self.args2kwargs(self._map_values)
        self._samples = samples

    def set_posterior_covariance(self, cov_matrix, num_samples=1000, seed=None):
        self._cov_matrix = cov_matrix
        self._samples = model_util.draw_samples_from_covariance(self.best_fit_values(),
                                                                self._cov_matrix,
                                                                num_samples=num_samples, 
                                                                seed=seed)

    def update_fixed(self, kwargs_fixed, kwargs_prior=None):
        # TODO: fill current and init values with values that were previously fixed, if needed
        self._set_params_update_fixed(kwargs_fixed, 'mass_model_list', 'kwargs_lens')
        self._set_params_update_fixed(kwargs_fixed, 'source_model_list', 'kwargs_source')
        self._set_params_update_fixed(kwargs_fixed, 'lens_light_model_list', 'kwargs_lens_light')

        # update fixed settings and everything that depends on it
        self._kwargs_fixed.update(kwargs_fixed)
        if kwargs_prior is not None:
            self._kwargs_prior = kwargs_prior
        self._update_arrays()

    # @partial(jit, static_argnums=(0,))
    def args2kwargs(self, args):
        i = 0
        args = jnp.atleast_1d(args)
        kwargs_lens, i = self._get_params(args, i, 'mass_model_list', 'kwargs_lens')
        kwargs_source, i = self._get_params(args, i, 'source_model_list', 'kwargs_source')
        kwargs_lens_light, i = self._get_params(args, i, 'lens_light_model_list', 'kwargs_lens_light')
        # apply joint param rules
        kwargs_lens = self._join_params(kwargs_lens, kwargs_lens, self._kwargs_joint['lens_with_lens'])
        kwargs_source = self._join_params(kwargs_source, kwargs_source, self._kwargs_joint['source_with_source'])
        kwargs_lens_light = self._join_params(kwargs_lens_light, kwargs_lens_light, self._kwargs_joint['lens_light_with_lens_light'])
        kwargs_lens = self._join_params(kwargs_lens_light, kwargs_lens, self._kwargs_joint['lens_with_lens_light'])
        # wrap-up
        kwargs = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source, 'kwargs_lens_light': kwargs_lens_light}
        return kwargs

    # @partial(jit, static_argnums=(0,))
    def kwargs2args(self, kwargs):
        args = self._set_params(kwargs, 'mass_model_list', 'kwargs_lens')
        args += self._set_params(kwargs, 'source_model_list', 'kwargs_source')
        args += self._set_params(kwargs, 'lens_light_model_list', 'kwargs_lens_light')
        return jnp.array(args)

    def kwargs2args_prior(self, kwargs_prior):
        types_m, lowers_m, uppers_m, means_m, widths_m = self._set_params_prior(kwargs_prior, 'mass_model_list', 'kwargs_lens')
        types_s, lowers_s, uppers_s, means_s, widths_s = self._set_params_prior(kwargs_prior, 'source_model_list', 'kwargs_source')
        types_l, lowers_l, uppers_l, means_l, widths_l = self._set_params_prior(kwargs_prior, 'lens_light_model_list', 'kwargs_lens_light')
        types =  types_m  + types_s  + types_l
        lowers = lowers_m + lowers_s + lowers_l
        uppers = uppers_m + uppers_s + uppers_l
        means  = means_m  + means_s  + means_l
        widths = widths_m + widths_s + widths_l
        return types, np.array(lowers), np.array(uppers), np.array(means), np.array(widths)

    def log_prior(self, args):
        logP = 0
        for i in range(self.num_parameters):
            gaussian_prior = self._prior_types[i] == 'gaussian'
            uniform_prior  = self._prior_types[i] == 'uniform'
            logP += lax.cond(gaussian_prior, lambda _: - 0.5 * ((args[i] - self._means[i]) / self._widths[i]) ** 2, lambda _: 0., operand=None)
            logP += lax.cond(uniform_prior, lambda _: lax.cond(args[i] < self._lowers[i], lambda _: - self._unif_prior_penalty, lambda _: 0., operand=None), lambda _: 0., operand=None)
            logP += lax.cond(uniform_prior, lambda _: lax.cond(args[i] > self._uppers[i], lambda _: - self._unif_prior_penalty, lambda _: 0., operand=None), lambda _: 0., operand=None)
        return logP

    def log_prior_gaussian(self, args):
        logP = 0
        for i in range(self.num_parameters):
            gaussian_prior = self._prior_types[i] == 'gaussian'
            logP += lax.cond(gaussian_prior, lambda _: - 0.5 * ((args[i] - self._means[i]) / self._widths[i]) ** 2, lambda _: 0., operand=None)
        return logP

    def log_prior_uniform(self, args):
        logP = 0.
        for i in range(self.num_parameters):
            uniform_prior  = self._prior_types[i] == 'uniform'
            # logP += lax.cond(uniform_prior, lambda _: lax.cond(args[i] < self._lowers[i], lambda _: - self._unif_prior_penalty, lambda _: 0., operand=None), lambda _: 0., operand=None)
            # logP += lax.cond(uniform_prior, lambda _: lax.cond(args[i] > self._uppers[i], lambda _: - self._unif_prior_penalty, lambda _: 0., operand=None), lambda _: 0., operand=None)
            logP += lax.cond(uniform_prior, lambda _: - (args[i] - jnp.clip(args[i], a_min=self._lowers[i], a_max=self._uppers[i]))**2, lambda _: 0., operand=None)
        return logP

    def apply_bounds(self, args):
        return jnp.clip(args, a_min=self._lowers, a_max=self._uppers)

    def log_prior_nojit(self, args):
        logP = 0
        for i in range(self.num_parameters):
            if self._prior_types[i] == 'gaussian':
                logP += - 0.5 * ((args[i] - self._means[i]) / self._widths[i]) ** 2
            elif self._prior_types[i] == 'uniform' and not (self._lowers[i] <= args[i] <= self._uppers[i]):
                logP += - self._unif_prior_penalty
        return logP

    @staticmethod
    def get_class_for_model(kwargs_key, model):
        # TODO: move outside of this class
        profile_class = None
        if kwargs_key in ['kwargs_source', 'kwargs_lens_light']:
            if model not in LIGHT_MODELS:
                raise ValueError(f"'{model}' is not supported.")
            if model == 'GAUSSIAN':
                profile_class = gaussian.Gaussian
            if model == 'GAUSSIAN_ELLIPSE':
                profile_class = gaussian.GaussianEllipse
            elif model == 'SERSIC':
                profile_class = sersic.Sersic
            elif model == 'SERSIC_ELLIPSE':
                profile_class = sersic.SersicElliptic
            elif model == 'CORE_SERSIC':
                profile_class = sersic.CoreSersic
            elif model == 'UNIFORM':
                profile_class = uniform.Uniform
            elif model == 'PIXELATED':
                profile_class = pixelated_light.Pixelated
            elif model == 'SHAPELETS':
                profile_class = shapelets.Shapelets
        elif kwargs_key == 'kwargs_lens':
            if model not in MASS_MODELS:
                raise ValueError("'{model}' is not supported.")
            elif model == 'GAUSSIAN':
                profile_class = gaussian_potential.Gaussian
            elif model == 'EPL':
                profile_class = epl.EPL
            elif model == 'SIE':
                profile_class = sie.SIE
            elif model == 'SIS':
                profile_class = sis.SIS
            elif model == 'NIE':
                profile_class = nie.NIE
            elif model == 'POINT_MASS':
                profile_class = point_mass.PointMass
            elif model == 'SHEAR':
                profile_class = shear.Shear
            elif model == 'SHEAR_GAMMA_PSI':
                profile_class = shear.ShearGammaPsi
            elif model == 'MULTIPOLE':
                profile_class = multipole.Multipole
            elif model == 'PIXELATED':
                profile_class = pixelated_lens.PixelatedPotential
            elif model == 'PIXELATED_DIRAC':
                profile_class = pixelated_lens.PixelatedPotentialDirac
        if profile_class is None:
            raise ValueError(f"Could not find the model class for '{model}'")
        return profile_class

    @staticmethod
    def get_param_names_for_model(kwargs_key, model):
        return Parameters.get_class_for_model(kwargs_key, model).param_names

    def _update_arrays(self):
        self._kwargs_fixed = self._update_fixed_with_joint(self._kwargs_fixed, self._kwargs_joint)
        self._prior_types, self._lowers, self._uppers, self._means, self._widths \
            = self.kwargs2args_prior(self._kwargs_prior)
        self._init_values = self.kwargs2args(self._kwargs_init)
        self._kwargs_init = self.args2kwargs(self._init_values)  # for updating missing fields
        self._num_params = len(self._init_values)
        if self.optimized:
            self._map_values = self.kwargs2args(self._kwargs_map)
        if hasattr(self, '_names'):
            delattr(self, '_names')
        if hasattr(self, '_symbols'):
            delattr(self, '_symbols')
        if hasattr(self, '_samples'):
            delattr(self, '_samples')
        if hasattr(self, '_kwargs_samples'):
            delattr(self, '_kwargs_samples')

    def _update_fixed_with_joint(self, kwargs_fixed, kwargs_joint):
        kwargs_fixed = self._update_fixed_with_joint_one(kwargs_fixed, kwargs_joint, 'kwargs_lens', 'lens_with_lens')
        kwargs_fixed = self._update_fixed_with_joint_one(kwargs_fixed, kwargs_joint, 'kwargs_source', 'source_with_source')
        kwargs_fixed = self._update_fixed_with_joint_one(kwargs_fixed, kwargs_joint, 'kwargs_lens_light', 'lens_light_with_lens_light')
        kwargs_fixed = self._update_fixed_with_joint_one(kwargs_fixed, kwargs_joint, 'kwargs_lens', 'lens_with_lens_light')
        return kwargs_fixed

    @staticmethod
    def _update_fixed_with_joint_one(kwargs_fixed, kwargs_joint, kwargs_key, joint_key):
        kwargs_fixed_updt = deepcopy(kwargs_fixed)
        joint_setting_list = kwargs_joint[joint_key]
        for setting in joint_setting_list:
            (i_1, k_2), param_list = setting
            for param_names in param_list:
                if isinstance(param_names, (tuple, list)) and len(param_names) > 1:
                    param_name_1, param_name_2 = param_names
                else:
                    param_name_1 = param_name_2 = param_names
                kwargs_fixed_updt[kwargs_key][k_2][param_name_2] = 0
        return kwargs_fixed_updt

    @staticmethod
    def _join_params(kwargs_list_1, kwargs_list_2, joint_setting_list):
        for setting in joint_setting_list:
            (i_1, k_2), param_list = setting
            for param_names in param_list:
                if isinstance(param_names, (tuple, list)) and len(param_names) > 1:
                    param_name_1, param_name_2 = param_names
                else:
                    param_name_1 = param_name_2 = param_names
                kwargs_list_2[k_2][param_name_2] = kwargs_list_1[i_1][param_name_1]
        return kwargs_list_2

    def _get_params(self, args, i, kwargs_model_key, kwargs_key):
        kwargs_list = []
        for k, model in enumerate(self.kwargs_model[kwargs_model_key]):
            kwargs = {}
            kwargs_fixed_k = self._kwargs_fixed[kwargs_key][k]
            param_names = self.get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed_k:
                    if model == 'PIXELATED':
                        if kwargs_key == 'kwargs_lens':
                            n_pix_x, n_pix_y = self._image.MassModel.pixelated_shape
                        elif kwargs_key == 'kwargs_source':
                            n_pix_x, n_pix_y = self._image.SourceModel.pixelated_shape
                        elif kwargs_key == 'kwargs_lens_light':
                            n_pix_x, n_pix_y = self._image.LensLightModel.pixelated_shape
                        num_param = int(n_pix_x * n_pix_y)
                        kwargs['pixels'] = args[i:i + num_param].reshape(n_pix_x, n_pix_y)
                    elif model == 'SHAPELETS' and name == 'amps':
                        if kwargs_key == 'kwargs_source':
                            num_param = self._image.SourceModel.num_amplitudes_list[k]
                        elif kwargs_key == 'kwargs_lens_light':
                            num_param = self._image.LensLightModel.num_amplitudes_list[k]
                        else:
                            raise ValueError("Basis functions can only be in the source or lens light.")
                        kwargs['amps'] = args[i:i + num_param]
                    else:
                        num_param = 1
                        kwargs[name] = args[i]
                    i += num_param
                else:
                    kwargs[name] = kwargs_fixed_k[name]
            kwargs_list.append(kwargs)
        return kwargs_list, i

    def _set_params(self, kwargs, kwargs_model_key, kwargs_key):
        args = []
        for k, model in enumerate(self.kwargs_model[kwargs_model_key]):
            kwargs_profile = kwargs[kwargs_key][k]
            kwargs_fixed_k = self._kwargs_fixed[kwargs_key][k]
            param_names = self.get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed_k:
                    if model == 'PIXELATED':
                        pixels = kwargs_profile['pixels']
                        if kwargs_key == 'kwargs_lens':
                            n_pix_x, n_pix_y = self._image.MassModel.pixelated_shape
                        elif kwargs_key == 'kwargs_source':
                            n_pix_x, n_pix_y = self._image.SourceModel.pixelated_shape
                        elif kwargs_key == 'kwargs_lens_light':
                            n_pix_x, n_pix_y = self._image.LensLightModel.pixelated_shape
                        if isinstance(pixels, (int, float)):
                            pixels = pixels * np.ones((n_pix_x, n_pix_y))
                        elif pixels.shape != (n_pix_x, n_pix_y):
                            raise ValueError("Pixelated array is inconsistent with pixelated grid.")
                        args += pixels.flatten().tolist()
                    elif model == 'SHAPELETS' and name == 'amps':
                        amps = kwargs_profile['amps']
                        if kwargs_key == 'kwargs_source':
                            num_param = self._image.SourceModel.num_amplitudes_list[k]
                        elif kwargs_key == 'kwargs_lens_light':
                            num_param = self._image.LensLightModel.num_amplitudes_list[k]
                        else:
                            raise ValueError("Basis functions can only be in the source or lens light.")
                        if len(amps) != num_param:
                            raise ValueError("Number of functions' amplitudes is not the on expected.")
                        args += np.asarray(amps).flatten().tolist()
                    else:
                        args.append(kwargs_profile[name])
        return args

    def _set_params_prior(self, kwargs, kwargs_model_key, kwargs_key):
        types, lowers, uppers, means, widths = [], [], [], [], []
        for k, model in enumerate(self.kwargs_model[kwargs_model_key]):
            kwargs_profile = kwargs[kwargs_key][k]
            kwargs_fixed_k = self._kwargs_fixed[kwargs_key][k]
            param_names = self.get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed_k:
                    if name not in kwargs_profile:
                        prior_type = None
                    else:
                        prior_type = kwargs_profile[name][0]

                    if model == 'PIXELATED':
                        if kwargs_key == 'kwargs_lens':
                            n_pix_x, n_pix_y = self._image.MassModel.pixelated_shape
                        elif kwargs_key == 'kwargs_source':
                            n_pix_x, n_pix_y = self._image.SourceModel.pixelated_shape
                        elif kwargs_key == 'kwargs_lens_light':
                            n_pix_x, n_pix_y = self._image.LensLightModel.pixelated_shape
                        num_param = int(n_pix_x * n_pix_y)
                    elif model == 'SHAPELETS' and name == 'amps':
                        if kwargs_key == 'kwargs_source':
                            num_param = self._image.SourceModel.num_amplitudes_list[k]
                        elif kwargs_key == 'kwargs_lens_light':
                            num_param = self._image.LensLightModel.num_amplitudes_list[k]
                    else:
                        num_param = 1
                    
                    if prior_type == 'uniform':
                        if num_param > 1:
                            types += [prior_type]*num_param
                            lowers_tmp, uppers_tmp = kwargs_profile['pixels'][1], kwargs_profile['pixels'][2]
                            # those bounds can either be whole array (values per pixel)
                            if isinstance(lowers_tmp, (np.ndarray, jnp.ndarray)):
                                lowers += lowers_tmp.flatten().tolist()
                                uppers += uppers_tmp.flatten().tolist()
                            # or they can be single numbers, in which case they are considered the same for pixel
                            elif isinstance(lowers_tmp, (int, float)):
                                lowers += [float(lowers_tmp)]*num_param
                                uppers += [float(uppers_tmp)]*num_param
                            means  += [np.nan]*num_param
                            widths += [np.nan]*num_param
                        else:
                            types.append(prior_type)
                            lowers.append(float(kwargs_profile[name][1]))
                            uppers.append(float(kwargs_profile[name][2]))
                            means.append(np.nan)
                            widths.append(np.nan)

                    elif prior_type == 'gaussian':
                        if model in ['PIXELATED', 'SHAPELETS']:
                            raise ValueError(f"'gaussian' prior for '{model}' model is not supported")
                        else:
                            types.append(prior_type)
                            lowers.append(-np.inf)
                            uppers.append(+np.inf)
                            means.append(kwargs_profile[name][1])
                            widths.append(kwargs_profile[name][2])

                    else:
                        types  += [prior_type]*num_param
                        lowers += [-np.inf]*num_param
                        uppers += [+np.inf]*num_param
                        means  += [np.nan]*num_param
                        widths += [np.nan]*num_param

        return types, lowers, uppers, means, widths

    def _set_params_update_fixed(self, kwargs_fixed, kwargs_model_key, kwargs_key):
        for k, model in enumerate(self.kwargs_model[kwargs_model_key]):
            kwargs_fixed_k_old = self._kwargs_fixed[kwargs_key][k]
            kwargs_fixed_k_new = kwargs_fixed
            param_names = self.get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if name in kwargs_fixed_k_old and name not in kwargs_fixed_k_new:
                    self._kwargs_init[kwargs_key][k][name] = deepcopy(kwargs_fixed_k_old[name])
                    if self.optimized:
                        self._kwargs_map[kwargs_key][k][name] = deepcopy(kwargs_fixed_k_old[name])

    @partial(jit, static_argnums=(0,))
    def _group_kwargs_samples_by_param(self, kwargs_samples):
        kwargs_samples_grouped = {}
        num_samples = len(kwargs_samples)
        for kwargs_model_key, kwargs_model in kwargs_samples[0].items():
            kwargs_samples_grouped[kwargs_model_key] = []
            for k in range(len(kwargs_model)):
                kwargs_samples_grouped[kwargs_model_key].append({})
                for param_name in kwargs_samples[0][kwargs_model_key][k].keys():
                    samples = []
                    for i in range(num_samples):
                        param_values_i = kwargs_samples[i][kwargs_model_key][k][param_name]
                        samples.append(param_values_i)
                    kwargs_samples_grouped[kwargs_model_key][k][param_name] = samples
        return kwargs_samples_grouped

    def _set_names(self, kwargs_model_key, kwargs_key):
        names = []
        short_id = kwargs_key.replace('kwargs_', '')
        for k, model in enumerate(self.kwargs_model[kwargs_model_key]):
            kwargs_fixed_k = self._kwargs_fixed[kwargs_key][k]
            param_names = self.get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed_k:
                    if model == 'PIXELATED':
                        if kwargs_key == 'kwargs_lens':
                            n_pix_x, n_pix_y = self._image.MassModel.pixelated_shape
                            num_param = int(n_pix_x * n_pix_y)
                            names_k = [f"psi_{i}" for i in range(num_param)]  # 'psi' for deflector
                        elif kwargs_key == 'kwargs_source':
                            n_pix_x, n_pix_y = self._image.SourceModel.pixelated_shape
                            num_param = int(n_pix_x * n_pix_y)
                            names_k = [f"s_{i}" for i in range(num_param)]  # 's' for source
                        elif kwargs_key == 'kwargs_lens_light':
                            n_pix_x, n_pix_y = self._image.LensLightModel.pixelated_shape
                            num_param = int(n_pix_x * n_pix_y)
                            names_k = [f"l_{i}" for i in range(num_param)]  # 'l' for potential
                    elif model == 'SHAPELETS' and name == 'amps':
                        if kwargs_key == 'kwargs_source':
                            num_param = self._image.SourceModel.num_amplitudes_list[k]
                        elif kwargs_key == 'kwargs_lens':
                            num_param = self._image.LensLightModel.num_amplitudes_list[k]
                        names_k = [f"amp_{i}" for i in range(num_param)]
                    else:
                        names_k = [name]
                    names += [f"{n}-{short_id}-{k}" for n in names_k]  # assign a unique identifier
        return names

    @staticmethod
    def name2latex(name_raw):
        # TODO: move outside of this class
        name, model_type, profile_idx = name_raw.split('-')   # encapsulate this line in a well-named method

        # pixelated models
        if name[:2] == 's_':  
            latex = r"$s_{" + r"{}".format(int(name[2:])) + r"}$"
        elif name[:2] == 'l_':  
            latex = r"$l_{" + r"{}".format(int(name[2:])) + r"}$"
        elif name[:4] == 'psi_':  
            latex = r"$\psi_{" + r"{}".format(int(name[4:])) + r"}$"

        # analytical models
        elif name == 'theta_E':
            latex = r"$\theta_{\rm E}$"
        elif name == 'gamma':
            latex = r"$\gamma'$"
        elif name == 'gamma_ext':
            latex = r"$\gamma_{\rm ext}$"
        elif name == 'psi_ext':
            latex = r"$\psi_{\rm ext}$"
        elif name == 'gamma1':
            latex = r"$\gamma_{\rm 1, ext}$"
        elif name == 'gamma2':
            latex = r"$\gamma_{\rm 2, ext}$"
        elif name == 'amp':
            latex = r"$A$"
        elif name == 'R_sersic':
            latex = r"$R_{\rm Sersic}$"
        elif name == 'n_sersic':
            latex = r"$n_{\rm Sersic}$"
        elif name == 'e1':
            latex = r"$e_1$"
        elif name == 'e2':
            latex = r"$e_2$"
        elif name == 'center_x':
            latex = r"$x_0$"
        elif name == 'center_y':
            latex = r"$y_0$"
        elif name == 'ra_0':
            latex = r"${\rm RA}_0$"
        elif name == 'dec_0':
            latex = r"${\rm Dec}_0$"
        elif name == 'm':
            latex = r"$m$"
        elif name == 'a_m':
            latex = r"$a_m$"
        elif name == 'phi_m':
            latex = r"$\phi_m$"
        else:
            raise ValueError("latex symbol for variable '{}' is unknown".format(name))
        return latex

    def _name2latex(self, names):
        latexs = []
        for name in names:
            latexs.append(self.name2latex(name))
        return latexs
