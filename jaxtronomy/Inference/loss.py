from functools import partial
from jax import jit
import jax.numpy as jnp


class Loss(object):
    """Class that manages the loss function, defined as -[log(likelihood) + log(prior)]"""

    def __init__(self, data, image_class, param_class, 
                 likelihood_type='gaussian', 
                 regularization_terms=['starlets_l1']):
        self._data  = data
        self._image = image_class
        self._param = param_class
        if likelihood_type == 'gaussian':
            self._log_likelihood = self._gaussian_log_likelihood
        else:
            raise NotImplementedError("Likelihood term '{}' ")
    
    @partial(jit, static_argnums=(0,))
    def __call__(self, args):
        return self.loss(args)

    @partial(jit, static_argnums=(0,))
    def loss(self, args):
        model = self._image.model(**self._param.args2kwargs(args))
        log_L = self.log_likelihood(model)
        log_P = self.log_prior(args)
        return - log_L - log_P

    @partial(jit, static_argnums=(0,))
    def loss_kwargs(self, kwargs):
        model = self._image.model(**kwargs)
        log_L = self.log_likelihood(model)
        log_P = self.log_prior(self._param.kwargs2args(kwargs))
        return - log_L - log_P

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, model):
        return self._log_likelihood(model)

    @partial(jit, static_argnums=(0,))
    def log_prior(self, args):
        return self._param.log_prior_no_uniform(args)

    @partial(jit, static_argnums=(0,))
    def _gaussian_log_likelihood(self, model):
        #noise_var = self._image.C_D_model(model)
        noise_var = self._image.Noise.C_D
        return - 0.5 * jnp.sum((self._data - model)**2 / noise_var)
