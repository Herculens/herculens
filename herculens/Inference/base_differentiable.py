# Defines a general fully differentiable scalar function
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


from functools import partial
from jax import jit, grad, jacfwd, jacrev, jvp, value_and_grad


__all__ = ['Differentiable']


class Differentiable(object):

    """Abstract class that defines a function with its derivatives, typically the loss function.
    """

    def _func(self, args):
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def __call__(self, args):
        """alias differentiable function"""
        return self._func(args)

    @partial(jit, static_argnums=(0,))
    def function(self, args):
        return self._func(args)

    @partial(jit, static_argnums=(0,))
    def gradient(self, args):
        """gradient (first derivative) of the loss function"""
        return grad(self._func)(args)

    @partial(jit, static_argnums=(0,))
    def value_and_gradient(self, args):
        return value_and_grad(self._func)(args)

    @partial(jit, static_argnums=(0,))
    def hessian(self, args):
        """hessian (second derivative) of the loss function"""
        return jacfwd(jacrev(self._func))(args)

    @partial(jit, static_argnums=(0,))
    def hessian_vec_prod(self, args, vec):
        """hessian-vector product"""
        # forward-over-reverse (https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode)
        return jvp(grad(self._func), (args,), (vec,))[1]
