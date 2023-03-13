# Defines the model of a strong lens
# 
# Copyright (c) 2022, herculens developers and contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'aymgal'


import jax
import numpyro
from numpyro import handlers
from numpyro.infer import util
from numpyro.distributions.transforms import biject_to

from herculens.Inference.ProbModel.base_model import BaseProbModel


__all__ = ['NumpyroModel']


def unconstrain_fn(model, model_args, model_kwargs, params):
    """
    Transforms parameter values from constrained to unconstrained space.
    This function performs the inverse transform from numpyro.infer.util.constrain_fn().
    """
    substituted_model = handlers.substitute(model, params)
    model_trace = handlers.trace(substituted_model).get_trace(*model_args, **model_kwargs)
    constrained_values, inv_transforms = {}, {}
    for k, v in model_trace.items():
        if (
            v["type"] == "sample"
            and not v["is_observed"]
            and not v["fn"].support.is_discrete
        ):
            constrained_values[k] = v["value"]
            with util.helpful_support_errors(v):
                inv_transforms[k] = biject_to(v["fn"].support)
    params_const = util.transform_fn(
        inv_transforms,
        {k: v for k, v in constrained_values.items()},
        invert=True,
    )
    return params_const


class NumpyroModel(BaseProbModel):
    """Defines a numpyro model based on a LensImage instance"""

    @property
    def num_parameters(self):
        if not hasattr(self, '_num_param'):
            num_param = 0
            for site in self.get_trace().values():
                if (site['type'] == 'sample' and not site['is_observed']
                    or site['type'] == 'param'):
                    num_param += site['value'].size
            self._num_param = num_param
        return self._num_param

    def log_prob(self, params, constrained=True):
        """returns the logarithm of the data likelihood plus the logarithm of the prior"""
        if constrained is True:
            # do this for optimisation in constrained space
            log_prob, model_trace = util.log_density(self.model, (), {}, params)
        else:
            # do this for optimisation in unconstrained space
            log_prob = - util.potential_energy(self.model, (), {}, params)
        return log_prob
    
    def log_likelihood(self, params):
        # returns the logarithm of the data likelihood
        return util.log_likelihood(self.model, params, batch_ndims=0)['obs']

    def seeded_model(self, seed=0):
        return handlers.seed(self.model, jax.random.PRNGKey(seed))
    
    def get_trace(self, seed=0):
        return handlers.trace(self.seeded_model(seed=seed)).get_trace()

    def get_sample(self, seed=0):
        trace = self.get_trace(seed=seed)
        return {site['name']: site['value'] for site in trace.values() if not site.get('is_observed', False)}

    def sample_prior(self, num_samples, seed=0):
        batch_ndims = 0 if num_samples else 1
        predictive = util.Predictive(self.model, 
                                     num_samples=num_samples, 
                                     batch_ndims=batch_ndims)
        samples = predictive(jax.random.PRNGKey(seed))
        del samples['obs']
        return samples

    def render_model(self):
        return numpyro.render_model(self.model)

    def params2kwargs(self, params, transform=False, constrained=True):
        """
        If transform=True, parameters will get transformed to their codomain.
        If constrained=True, assumes that the input are in constrained space,
        otherwise they are assumed to be in constrained values
        """
        if transform is False:
            params_ = params
        elif constrained is True:
            params_ = self.unconstrain(params)
        else:
            params_ = self.constrain(params)
        return self._params2kwargs(params_)

    def constrain(self, params):
        return util.constrain_fn(self.model, (), {}, params)

    def unconstrain(self, params):
        return unconstrain_fn(self.model, (), {}, params)
