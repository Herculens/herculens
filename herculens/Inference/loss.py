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

    def __init__(self, prob_model, constrained_space=False, cap_value=None):
        """
        :param prob_model: probabilistic model (e.g. from numpyro) that has a
        log_prob() method that returns the full log-probability of the model
        :param constrained_space: whether or not to consider that the parameters
        (input values of log_prob()) are assumed to be in constrained or 
        unconstrained space
        """
        self._prob_model = prob_model
        self._constrained = constrained_space
        self._cap_value = cap_value

    def _func(self, args):
        """negative log-probability"""
        loss = - self._prob_model.log_prob(args, constrained=self._constrained)
        loss = jnp.nan_to_num(loss, nan=1e15, posinf=1e15, neginf=1e15)
        if self._cap_value is not None:
            loss = jnp.clip(loss, a_min=self._cap_value)
        return loss
