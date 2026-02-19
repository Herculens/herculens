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
from herculens.Util import numpyro_util


__all__ = ['NumpyroModel']


class NumpyroModel(BaseProbModel):
    """Defines a numpyro model based on a LensImage instance"""

    @property
    def num_parameters(self):
        """
        Returns the number of parameters in the model.
        It is advised to use the more general count_sampled_parameters() method. This property
        is only here for backward compatibility with older Herculens+Numpyro code.
        """
        if not hasattr(self, '_num_param'):
            try:
                self._num_param = self.count_sampled_parameters(model_args=(), model_kwargs={})
            except TypeError as e:
                print("Error while calling the property NumpyroModel.num_parameters."
                      "The cause might be that the underlying numpyro model requires"
                      "specific positional and/or keyword arguments."
                      "Use the count_sampled_parameters(model_args=(), model_kwargs={}) method instead.\n"
                      f"Here is the original error:\n{e}")
        return self._num_param
    
    def count_sampled_parameters(self, model_args=(), model_kwargs={}):
        return numpyro_util.count_sampled_parameters(self.model, model_args=model_args, model_kwargs=model_kwargs)
        
    def log_prob(self, params, constrained=False, model_args=(), model_kwargs={}):
        """returns the logarithm of the data likelihood plus the logarithm of the prior"""
        if constrained is True:
            # do this for optimisation in constrained space
            log_prob, model_trace = util.log_density(self.model, model_args, model_kwargs, params)
        else:
            # do this for optimisation in unconstrained space
            # TODO: use the new numpyro function potential_fn instead
            log_prob = - numpyro_util.potential_energy(self.model, model_args, model_kwargs, params)
        return log_prob
    
    def log_likelihood(self, params, obs_site_key='obs', model_args=(), model_kwargs={}):
        # returns the logarithm of the data likelihood
        return util.log_likelihood(self.model, params, batch_ndims=0, *model_args, **model_kwargs)[obs_site_key]

    def seeded_model(self, prng_key):
        return handlers.seed(self.model, prng_key)
    
    def get_trace(self, prng_key, model_args=(), model_kwargs={}):
        return handlers.trace(self.seeded_model(prng_key)).get_trace(*model_args, **model_kwargs)

    def get_sample(self, prng_key=None, model_args=(), model_kwargs={}):
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        trace = self.get_trace(prng_key, model_args=model_args, model_kwargs=model_kwargs)
        return {site['name']: site['value'] for site in trace.values() if not site.get('is_observed', False)}

    def sample_prior(self, num_samples, prng_key=None, model_args=(), model_kwargs={}):
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        batch_ndims = 0 if num_samples else 1
        predictive = util.Predictive(self.model, 
                                     num_samples=num_samples, 
                                     batch_ndims=batch_ndims)
        samples = predictive(prng_key, *model_args, **model_kwargs)
        # delete all sites whose key contain 'obs'
        sites_keys = copy.deepcopy(list(samples.keys()))
        for key in sites_keys:
            if 'obs' in key:
                del samples[key]
        return samples

    def render_model(self, model_args=(), model_kwargs={}):
        return numpyro.render_model(self.model, model_args=model_args, model_kwargs=model_kwargs)

    def constrain(self, params, model_args=(), model_kwargs={}):
        return util.constrain_fn(self.model, model_args, model_kwargs, params)

    def unconstrain(self, params, model_args=(), model_kwargs={}):
        return util.unconstrain_fn(self.model, model_args, model_kwargs, params)
