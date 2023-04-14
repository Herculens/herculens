# Defines a general fully differentiable probability function for inference purposes
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


from functools import partial
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from tqdm import tqdm


__all__ = ['BaseOptimizer']


class BaseOptimizer(object):

    """Abstract class that defines wraps the loss function for use with optimizers.
    :param loss: herculens.Inference.loss.Loss instance
    :param loss_norm_optim: normalization constant for reducing the magnitude of the loss during optimization.
    """

    def __init__(self, loss, loss_norm_optim=1.):
        self.loss = loss
        self.norm_optim = loss_norm_optim

    def run(self, *args, **kwargs):
        raise NotImplementedError("`run()` method must be implemented.")

    @partial(jit, static_argnums=(0,))
    def function_optim(self, args):
        return self.loss(args) / self.norm_optim

    @partial(jit, static_argnums=(0,))
    def grad_function_optim(self, args):
        return grad(self.loss)(args)

    @partial(jit, static_argnums=(0,))
    def function_optim_with_grad(self, args):
        return value_and_grad(self.function_optim)(args)

    @partial(jit, static_argnums=(0,))
    def function_optim_LM_scalar(self, args):
        return jnp.array([self.function_optim(args)])
    
    @staticmethod
    def _for_loop(iterable, progress_bar_bool, **tqdm_kwargs):
        if progress_bar_bool is True:
            return tqdm(iterable, **tqdm_kwargs)
        else:
            return iterable
 