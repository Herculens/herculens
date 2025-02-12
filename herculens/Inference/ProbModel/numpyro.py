# Defines the model of a strong lens
# 
# Copyright (c) 2022, herculens developers and contributors


__author__ = 'aymgal'


import copy
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers
from numpyro.infer import util

from herculens.Inference.ProbModel.base_model import BaseProbModel
from herculens.Util import numpyro_util as my_util


__all__ = ['NumpyroModel']


class NumpyroModel(BaseProbModel):
    """Defines a numpyro model based on a LensImage instance"""

    @property
    def num_parameters(self):
        if not hasattr(self, '_num_param'):
            num_param = 0
            prng_key = jax.random.PRNGKey(0)  # seed not relevant
            for site in self.get_trace(prng_key).values():
                if (site['type'] == 'sample' and not site['is_observed']
                    or site['type'] == 'param'):
                    num_param += site['value'].size
            self._num_param = num_param
        return self._num_param

    def log_prob(self, params, constrained=False):
        """returns the logarithm of the data likelihood plus the logarithm of the prior"""
        if constrained is True:
            # do this for optimisation in constrained space
            log_prob, model_trace = util.log_density(self.model, (), {}, params)
        else:
            # do this for optimisation in unconstrained space
            log_prob = - my_util.potential_energy(self.model, (), {}, params)
        return log_prob
    
    def log_likelihood(self, params, obs_site_key='obs'):
        # returns the logarithm of the data likelihood
        return util.log_likelihood(self.model, params, batch_ndims=0)[obs_site_key]

    def seeded_model(self, prng_key):
        return handlers.seed(self.model, prng_key)
    
    def get_trace(self, prng_key):
        return handlers.trace(self.seeded_model(prng_key)).get_trace()

    def get_sample(self, prng_key=None):
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        trace = self.get_trace(prng_key)
        return {site['name']: site['value'] for site in trace.values() if not site.get('is_observed', False)}

    def sample_prior(self, num_samples, prng_key=None):
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        batch_ndims = 0 if num_samples else 1
        predictive = util.Predictive(self.model, 
                                     num_samples=num_samples, 
                                     batch_ndims=batch_ndims)
        samples = predictive(prng_key)
        # delete all sites whose key contain 'obs'
        sites_keys = copy.deepcopy(list(samples.keys()))
        for key in sites_keys:
            if 'obs' in key:
                del samples[key]
        return samples

    def render_model(self):
        return numpyro.render_model(self.model)

    def constrain(self, params):
        return util.constrain_fn(self.model, (), {}, params)

    def unconstrain(self, params):
        return util.unconstrain_fn(self.model, (), {}, params)
