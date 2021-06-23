import functools
import numpy as np
import jax.numpy as jnp
from jax import lax, jit


__all__ = ['Parameters']


class Parameters(object):
    """Class that manages parameters in JAX / auto-differentiable framework.
    Currently, it handles:
    - conversions from the differentiable parameter vector to user-friendly dictionnaries (args-to-kwargs, kwargs-to-args)
    - uniform and gaussian priors for subsets of parameters
    - log-prior values, that are meant to be added to the full loss function
    - nice LaTeX format for parameter names
    """

    _bound_penalty = 1e10

    def __init__(self, kwargs_model, kwargs_init, kwargs_prior, kwargs_fixed):
        self._kwargs_model = kwargs_model
        self._kwargs_init  = kwargs_init
        self._kwargs_prior = kwargs_prior
        self._kwargs_fixed = kwargs_fixed
        
        self._inits  = self.kwargs2args(self._kwargs_init)
        self._prior_types, self._lowers, self._uppers, self._means, self._widths \
            = self.kwargs2args_prior(self._kwargs_prior)

        self._num_params = len(self._inits)

        # TODO: write function that checks that no fields are missing
        # and fill those with default values if needed

    @property
    def num_parameters(self):
        return self._num_params

    @property
    def prior_types(self):
        return self._prior_types

    @property
    def bounds(self):
        return self._lowers, self._uppers

    @property
    def names(self):
        if not hasattr(self, '_names'):
            self._names = self._set_names('lens_model_list', 'kwargs_lens')
            self._names += self._set_names('source_model_list', 'kwargs_source')
            self._names += self._set_names('lens_light_model_list', 'kwargs_lens_light')
        return self._names

    @property
    def symbols(self):
        if not hasattr(self, '_symbols'):
            self._symbols = self._name2latex(self.names)
        return self._symbols

    def initial_values(self, as_kwargs=False, original=False):
        if hasattr(self, '_best_fit') and original is False:
            return self.best_fit_values(as_kwargs=as_kwargs)
        return self._kwargs_init if as_kwargs else self._inits
    
    def set_best_fit(self, args):
        self._best_fit = args
        if hasattr(self, '_kwargs_best_fit'):
            delattr(self, '_kwargs_best_fit')

    def best_fit_values(self, as_kwargs=False):
        if not hasattr(self, '_kwargs_best_fit') and as_kwargs is True:
            self._kwargs_best_fit = self.args2kwargs(self._best_fit)
        return self._kwargs_best_fit if as_kwargs else self._best_fit

    def set_samples(self, samples):
        self._map = jnp.median(samples, axis=0)  # maximum a-posterio values
        if hasattr(self, '_kwargs_map'):
            delattr(self, '_kwargs_map')

    def map_values(self, as_kwargs=False):
        if not hasattr(self, '_kwargs_map') and as_kwargs is True:
            self._kwargs_map = self.args2kwargs(self._map)
        return self._kwargs_map if as_kwargs else self._map

    #@functools.partial(jit, static_argnums=(0,))
    def args2kwargs(self, args):
        i = 0
        args = jnp.atleast_1d(args)
        kwargs_lens, i = self._get_params(args, i, 'lens_model_list', 'kwargs_lens')
        kwargs_source, i = self._get_params(args, i, 'source_model_list', 'kwargs_source')
        kwargs_lens_light, i = self._get_params(args, i, 'lens_light_model_list', 'kwargs_lens_light')
        kwargs = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source, 'kwargs_lens_light': kwargs_lens_light}
        return kwargs

    #@functools.partial(jit, static_argnums=(0,))
    def kwargs2args(self, kwargs):
        args = self._set_params(kwargs, 'lens_model_list', 'kwargs_lens')
        args += self._set_params(kwargs, 'source_model_list', 'kwargs_source')
        args += self._set_params(kwargs, 'lens_light_model_list', 'kwargs_lens_light')
        return jnp.array(args)

    def kwargs2args_prior(self, kwargs_prior):
        types_m, lowers_m, uppers_m, means_m, widths_m = self._set_params_prior(kwargs_prior, 'lens_model_list', 'kwargs_lens')
        types_s, lowers_s, uppers_s, means_s, widths_s = self._set_params_prior(kwargs_prior, 'source_model_list', 'kwargs_source')
        types_l, lowers_l, uppers_l, means_l, widths_l = self._set_params_prior(kwargs_prior, 'lens_light_model_list', 'kwargs_lens_light')
        types =  types_m  + types_s  + types_l
        lowers = lowers_m + lowers_s + lowers_l
        uppers = uppers_m + uppers_s + uppers_l
        means  = means_m  + means_s  + means_l
        widths = widths_m + widths_s + widths_l
        return types, np.array(lowers), np.array(uppers), np.array(means), np.array(widths)

    #@jit
    def log_prior(self, args):
        logP = 0
        for i in range(self.num_parameters):
            gaussian_prior = self._prior_types[i] == 'gaussian'
            uniform_prior  = self._prior_types[i] == 'uniform'
            logP += lax.cond(gaussian_prior, lambda _: - 0.5 * ((args[i] - self._means[i]) / self._widths[i]) ** 2, lambda _: 0., operand=None)
            logP += lax.cond(uniform_prior, lambda _: lax.cond(args[i] < self._lowers[i], lambda _: - self._bound_penalty, lambda _: 0., operand=None), lambda _: 0., operand=None)
            logP += lax.cond(uniform_prior, lambda _: lax.cond(args[i] > self._uppers[i], lambda _: - self._bound_penalty, lambda _: 0., operand=None), lambda _: 0., operand=None)
        return logP

    def _get_params(self, args, i, kwargs_model_key, kwargs_key):
        kwargs_list = []
        for k, model in enumerate(self._kwargs_model[kwargs_model_key]):
            kwargs = {}
            kwargs_fixed = self._kwargs_fixed[kwargs_key][k]
            param_names = self._get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed:
                    kwargs[name] = args[i]
                    i += 1
                else:
                    kwargs[name] = kwargs_fixed[name]
            kwargs_list.append(kwargs)
        return kwargs_list, i

    def _set_params(self, kwargs, kwargs_model_key, kwargs_key):
        args = []
        for k, model in enumerate(self._kwargs_model[kwargs_model_key]):
            kwargs_profile = kwargs[kwargs_key][k]
            kwargs_fixed = self._kwargs_fixed[kwargs_key][k]
            param_names = self._get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed:
                    args.append(kwargs_profile[name])
        return args

    def _set_params_prior(self, kwargs, kwargs_model_key, kwargs_key):
        types, lowers, uppers, means, widths = [], [], [], [], []
        for k, model in enumerate(self._kwargs_model[kwargs_model_key]):
            kwargs_profile = kwargs[kwargs_key][k]
            kwargs_fixed = self._kwargs_fixed[kwargs_key][k]
            param_names = self._get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed:
                    if name in kwargs_profile:
                        prior_type = kwargs_profile[name][0]
                        types.append(prior_type)
                        if prior_type == 'uniform':
                            lowers.append(kwargs_profile[name][1])
                            uppers.append(kwargs_profile[name][2])
                            means.append(np.nan)
                            widths.append(np.nan)
                        elif prior_type == 'gaussian':
                            lowers.append(np.nan)
                            uppers.append(np.nan)
                            means.append(kwargs_profile[name][1])
                            widths.append(kwargs_profile[name][2])
                        else:
                            types.append(None)
                            lowers.append(np.nan)
                            uppers.append(np.nan)
                            means.append(np.nan)
                            widths.append(np.nan)
                    else:
                        types.append(None)
                        lowers.append(np.nan)
                        uppers.append(np.nan)
                        means.append(np.nan)
                        widths.append(np.nan)
        return types, lowers, uppers, means, widths

    @staticmethod
    def _get_param_names_for_model(kwargs_key, model):
        if kwargs_key == 'kwargs_source':
            if model == 'GAUSSIAN':
                from jaxtronomy.LightModel.Profiles.gaussian import Gaussian
                profile_class = Gaussian
            elif model == 'SERSIC':
                from jaxtronomy.LightModel.Profiles.sersic import Sersic
                profile_class = Sersic
            elif model == 'PIXELATED':
                from jaxtronomy.LightModel.Profiles.pixelated import PixelatedSource
                profile_class = PixelatedSource
        elif kwargs_key == 'kwargs_lens':
            if model == 'SIE':
                from jaxtronomy.LensModel.Profiles.sie import SIE
                profile_class = SIE
            elif model == 'SHEAR':
                from jaxtronomy.LensModel.Profiles.shear import Shear
                profile_class = Shear
            elif model == 'SHEAR_GAMMA_PSI':
                from jaxtronomy.LensModel.Profiles.shear import ShearGammaPsi
                profile_class = Shear
        return profile_class.param_names

    def _set_names(self, kwargs_model_key, kwargs_key):
        names = []
        for k, model in enumerate(self._kwargs_model[kwargs_model_key]):
            kwargs_fixed = self._kwargs_fixed[kwargs_key][k]
            param_names = self._get_param_names_for_model(kwargs_key, model)
            for name in param_names:
                if not name in kwargs_fixed:
                    names.append(name)
        return names

    def _name2latex(self, names):
        latexs = []
        for name in names:
            if name == 'theta_E':
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
                latex = r"$c_{x,0}$"
            elif name == 'center_y':
                latex = r"$c_{y,0}$"
            else:
                raise ValueError("latex symbol for variable '{}' is unknown".format(name))
            latexs.append(latex)
        return latexs
