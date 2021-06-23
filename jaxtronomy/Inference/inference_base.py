from functools import partial
from jax import jit, grad, jacfwd, jacrev, jvp


__all__ = ['InferenceBase']


class InferenceBase(object):

    def __init__(self, loss_fn, param_class):
        self._loss_fn = loss_fn
        self._param_class = param_class

    @partial(jit, static_argnums=(0,))  # because first argument is 'self'
    def loss(self, args):
        return self._loss_fn(self._param_class.args2kwargs(args))

    @partial(jit, static_argnums=(0,))
    def jacobian(self, args):
        return grad(self.loss)(args)

    @partial(jit, static_argnums=(0,))
    def hessian(self, args):
        return jacfwd(jacrev(self.loss))(args)

    @partial(jit, static_argnums=(0,))
    def hessian_vec_prod(self, args, vec):
        # forward-over-reverse (https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode)
        return jvp(grad(self.loss), (args,), (vec,))[1]
