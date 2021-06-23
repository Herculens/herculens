from functools import partial
from jax import jit, grad, jacfwd, jacrev, jvp


__all__ = ['InferenceBase']


class InferenceBase(object):

    def __init__(self, loss_fn, param_class):
        self._loss_fn = loss_fn
        self._param_class = param_class

    @partial(jit, static_argnums=(0,))  # because first argument is 'self'
    def loss(self, args):
        """loss function to be minimized (aka negative log-likelihood + negative log-prior)"""
        return self._loss_fn(self._param_class.args2kwargs(args))

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
