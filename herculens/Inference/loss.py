# Defines the full loss function, from likelihood, prior and regularization terms
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal', 'austinpeel'


import numpy as np
import jax.numpy as jnp
from jax import jit
import warnings

from herculens.Inference.base_differentiable import Differentiable


__all__ = ['Loss']


class Loss(Differentiable):

    def __init__(self, prob_model, constrained_space=False):
        """
        :param prob_model: probabilistic model (e.g. from numpyro) that has a
        log_prob() method that returns the full log-probability of the model
        :param constrained_space: whether or not considering that parameters
        (input values of log_prob()) are assumed to be in constrained or 
        unconstrained space
        """
        self._prob_model = prob_model
        self._constrained = constrained_space

    def _func(self, args):
        """negative log-probability"""
        loss = - self._prob_model.log_prob(args, constrained=self._constrained)
        return jnp.nan_to_num(loss, nan=1e15, posinf=1e15, neginf=1e15)
