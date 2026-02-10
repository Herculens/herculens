# Defines the full loss function, from likelihood, prior and regularization terms
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal', 'austinpeel'


import types
import jax.numpy as jnp

from herculens.Inference.base_differentiable import Differentiable
from herculens.Inference.ProbModel.base_model import BaseProbModel


__all__ = ['Loss']


class Loss(Differentiable):

    def __init__(self, prob_model_or_log_prob, constrained_space=False, cap_value=None):
        """
        :param prob_model: probabilistic model (e.g. from numpyro) that has a
        log_prob() method that returns the full log-probability of the model
        :param constrained_space: whether or not to consider that the parameters
        (input values of log_prob()) are assumed to be in constrained or 
        unconstrained space
        """
        if isinstance(prob_model_or_log_prob, BaseProbModel):
            model = prob_model_or_log_prob
            self._log_prob = lambda args: model.log_prob(args, constrained=constrained_space)
        elif isinstance(prob_model_or_log_prob, types.FunctionType):
            self._log_prob = prob_model_or_log_prob
        else:
            raise TypeError("The first argument of Loss must be either a BaseProbModel instance "
                            "(e.g. herculens.Inference.ProbModel.numpyro.NumpyroModel) "
                            "or directly a function that returns the log-probability.")
        self._constrained = constrained_space
        self._cap_value = cap_value

    def _func(self, args):
        """negative log-probability"""
        loss = - self._log_prob(args)
        loss = jnp.nan_to_num(loss, nan=1e15, posinf=1e15, neginf=1e15)
        if self._cap_value is not None:
            loss = jnp.clip(loss, a_min=self._cap_value)
        return loss
