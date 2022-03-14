from functools import partial
from jax import jit, grad, jacfwd, jacrev, jvp


__all__ = ['InferenceBase']


class InferenceBase(object):

    """Class that defines wraps the loss function, and defins first and second order derivatives.
    :param loss_class: herculens.Inference.loss.Loss instance
    :param param_class: herculens.Parameters.parameters.Parameters instance
    """

    def __init__(self, loss_class, param_class):
        self._loss = loss_class
        self._param = param_class
        self.kinetic_fn = None  # for numpyro HMC, will default to Euclidean kinetic energy

    @property
    def parameters(self):
        return self._param

    @partial(jit, static_argnums=(0,))
    def loss(self, args):
        """
        loss function to be minimized
        Called if arguments of self._loss_fn should be args-like (i.e. as an array).
        """
        return self._loss(args)

    @partial(jit, static_argnums=(0,))
    def gradient(self, args):
        """gradient (first derivative) of the loss function"""
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
    def log_probability(self, args):
        """unnormalized log-posterior as the negative loss, typically for Bayesian inference using MCMC"""
        return - self.loss(args)

    @partial(jit, static_argnums=(0,))
    def potential_fn(self, args):
        """alias for negative log-likelihood, typically for Bayesian inference using HMC"""
        return self.loss(args)
