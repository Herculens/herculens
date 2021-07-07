from funct ools import partial
from jax import jit
import jax.numpy as jax


class Loss(object):
    """Class that manages the loss function, defined as -[log(likelihood) + log(prior)]"""

    def __init__(self, data_class, model_class, param_class, 
                 likelihood_terms=['gaussian'], 
                 regularization_terms=['starlets_l1']):
        self._data = data_class.data
        self._noise_var = data_class.noise_var  # TBD
        self._model_cls = model_class
        self._param_cls = param_class
        self._ll_terms  = likelihood_terms
        def _log_likelihood_tmp(model):
            log_L = 0.
            for ll_term in likelihood_terms:
                if ll_term == 'gaussian':
                    log_L += self._gaussian_log_likelihood(model, self._data, self._noise_var)
                else:
                    raise NotImplementedError("Likelihood term '{}' ")
            return log_L
        self._log_likelihood = _log_likelihood_tmp

    @partial(jit, static_argnums=(0,))
    def __call__(self, args):
        return self.loss(args)

    @partial(jit, static_argnums=(0,))
    def loss(self, args):
        model = self._model_cls.image(**self._param_cls.args2kwargs(args))
        log_L = self.log_likelihood(model, self._data, self._noise_var)
        log_P = self.log_prior(args)
        return - log_L - log_P

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, model):
        return self._log_likelihood(model)

    @partial(jit, static_argnums=(0,))
    def log_prior(self, args):
        return self._param_cls.log_prior(args)

    def _gaussian_log_likelihood(self, model, data, noise_var):
        noise_var = self._noise_var(model)
        return - 0.5 * jnp.sum((data - model)**2 / noise_var)
