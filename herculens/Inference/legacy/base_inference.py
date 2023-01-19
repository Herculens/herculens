# Defines a general fully differentiable probability function for inference purposes
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


from functools import partial
from jax import jit


__all__ = ['Inference']


class Inference(object):

    """Abstract class that defines wraps the loss function, and defines first and second order derivatives.
    :param loss_class: herculens.Inference.loss.Loss instance
    :param param_class: herculens.Parameters.parameters.Parameters instance
    """

    def __init__(self, loss_class, param_class):
        self._loss = loss_class
        self._param = param_class
        self.kinetic_fn = None  # for numpyro HMC, will default to Euclidean kinetic energy

    @partial(jit, static_argnums=(0,))
    def log_probability(self, args):
        """unnormalized log-posterior as the negative loss, typically for Bayesian inference using MCMC"""
        return - self._loss(args)

    @partial(jit, static_argnums=(0,))
    def potential_fn(self, args):
        """alias for negative log-likelihood, typically for Bayesian inference using HMC"""
        return self._loss(args)
        