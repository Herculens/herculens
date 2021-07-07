from functools import partial
from jax import jit, grad, jacfwd, jacrev, jvp


__all__ = ['InferenceBase']


class InferenceBase(object):

    def __init__(self, loss_class, param_class):
        self._loss = loss_class
        self._param = param_class
        self.kinetic_fn = None  # for numpyro HMC, will default to Euclidean kinetic energy

    @partial(jit, static_argnums=(0,))  # because first argument is 'self' and should be static
    def loss(self, args):
        """
        loss function to be minimized (aka negative log-likelihood + negative log-prior)
        Called if arguments of self._loss_fn should be args-like (i.e. as an array).
        """
        return self._loss(args)

    @partial(jit, static_argnums=(0,))
    def jacobian(self, args):
        """jacobian (first derivative) of the loss function"""
        return grad(self.loss)(args)

    @partial(jit, static_argnums=(0,))
    def hessian(self, args):
        """hessian (second derivative) of the loss function"""
        return jacfwd(jacrev(self.loss))(args)

    @partial(jit, static_argnums=(0,))
    def hessian_vec_prod(self, args, vec):
        """hessian-vector product"""
        # forward-over-reverse (https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode)
        return jvp(grad(self.loss), (args,), (vec,))[1]

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, args):
        """log-likelihood as the negative loss, typically for Bayesian inference using MCMC"""
        return - self.loss(args)

    @partial(jit, static_argnums=(0,))
    def potential_fn(self, args):
        """alias for negative log-likelihood, typically for Bayesian inference using HMC"""
        return self.loss(args)
