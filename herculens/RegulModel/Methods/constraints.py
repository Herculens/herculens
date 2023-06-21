# Defines regularization choices
# 
# Copyright (c) 2023, herculens developers and contributors


__author__ = 'aymgal'


import jax.numpy as jnp

from herculens.RegulModel.Methods.base import BaseRegularization


__all__ = [
    'Positivity',
    'Negativity',
]


class Positivity(BaseRegularization):

    param_names = ['strength']
    lower_limit_default = {'strength': 0}
    upper_limit_default = {'strength': 1e8}
    fixed_default = {key: True for key in param_names}

    def __init__(self, model_type, profile_index, mass_form=None):
        super().__init__(model_type, profile_index, mass_form=mass_form)

    def initialize(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def log_prob(self, kwargs_params, strength=0):
        pixels = self.get_pixel_params(kwargs_params)
        log_prob = - strength * jnp.abs(jnp.sum(jnp.minimum(0., pixels)))
        return log_prob


class Negativity(BaseRegularization):

    param_names = ['strength']
    lower_limit_default = {'strength': 0}
    upper_limit_default = {'strength': 1e8}
    fixed_default = {key: True for key in param_names}

    def __init__(self, model_type, profile_index, mass_form=None):
        super().__init__(model_type, profile_index, mass_form=mass_form)

    def initialize(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def log_prob(self, kwargs_params, strength=0):
        pixels = self.get_pixel_params(kwargs_params)
        log_prob = - strength * jnp.abs(jnp.sum(jnp.maximum(0., pixels)))
        return log_prob
