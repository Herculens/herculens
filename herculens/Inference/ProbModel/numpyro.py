# Defines the model of a strong lens
# 
# Copyright (c) 2022, herculens developers and contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'aymgal'


import jax
import numpyro
from numpyro import handlers
from numpyro.infer import util

from herculens.Inference.ProbModel.base_model import BaseProbModel


__all__ = ['NumpyroModel']


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

    def log_prob(self, params):
        # returns the logarithm of the data likelihood
        # plus the logarithm of the prior
        log_prob, trace = util.log_density(self.model, (), {}, params)
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