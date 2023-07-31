# Defines regularization choices
# 
# Copyright (c) 2023, herculens developers and contributors


__author__ = 'aymgal'


import jax.numpy as jnp

from herculens.RegulModel.Methods.base import BaseRegularization
from herculens.RegulModel import regul_util


__all__ = [
    'SparsityStarlet',
    'SparsityBLWavelet',
]


class BaseSparsityWaveletAnalysis(BaseRegularization):

    """
    Base class for regularization strategies imposing a sparsity constraint
    in transformed space (a.k.a. analysis framework)
    """

    def __init__(self, model_type, profile_index, mass_form=None):
        super().__init__(model_type, profile_index, mass_form=mass_form)
        self.transform = None
        self.weights = None
        self._transform_name = None
        self._second_gen = None

    def initialize(self, lens_image, kwargs_params, **kwargs_weights):
        if self.model_type in ('source', 'lens_light'):
            fn = regul_util.data_noise_to_wavelet_light
        elif self.model_type == 'lens_mass' and self._mass_form == 'potential':
            fn = regul_util.data_noise_to_wavelet_potential
        else:
            raise ValueError(f"Combination of {self.model_type} with "
                             f"lens mass '{self._mass_form}' is not supported")
        weights_list, transform_list = fn(lens_image, kwargs_params,
                                          model_type=self.model_type,
                                          wavelet_type_list=[self._transform_name],  # TODO: improve interface
                                          starlet_second_gen=self._second_gen,
                                          **kwargs_weights)
        self.transform = transform_list[0]
        self.weights = weights_list[0]


class SparsityStarlet(BaseSparsityWaveletAnalysis):

    param_names = ['lambda_0', 'lambda_1']
    lower_limit_default = {'lambda_0': 0, 'lambda_1': 0}
    upper_limit_default = {'lambda_0': 1e8, 'lambda_1': 1e8}
    fixed_default = {key: True for key in param_names}

    def __init__(self, model_type, profile_index, mass_form=None, second_gen=False):
        super().__init__(model_type, profile_index, mass_form=mass_form)
        self._transform_name = 'starlet'
        self._second_gen = second_gen

    def log_prob(self, kwargs_params, lambda_0=0, lambda_1=0):
        # get the pixels to be regularized
        pixels = self.get_pixel_params(kwargs_params)
        # pixels in transformed space
        coeffs = self.transform.decompose(pixels)
        # regularization weights
        W = self.weights
        # first scale (i.e. high frequencies)
        l1_weighted_coeffs0 = jnp.sum(jnp.abs(W[0] * coeffs[0]))
        # other scales (except coarsest)
        l1_weighted_coeffs1 = jnp.sum(jnp.abs(W[1:-1] * coeffs[1:-1]))
        # sum the two terms and minus sign to get log-probability
        log_prob = - lambda_0 * l1_weighted_coeffs0 - lambda_1 * l1_weighted_coeffs1
        return log_prob


class SparsityBLWavelet(BaseSparsityWaveletAnalysis):

    param_names = ['lambda_0']
    lower_limit_default = {'lambda_0': 0}
    upper_limit_default = {'lambda_0': 1e8}
    fixed_default = {key: True for key in param_names}

    def __init__(self, model_type, profile_index, mass_form=None):
        super().__init__(model_type, profile_index, mass_form=mass_form)
        self._transform_name = 'battle-lemarie-3'

    def log_prob(self, kwargs_params, lambda_0=0):
        # get the pixels to be regularized
        pixels = self.get_pixel_params(kwargs_params)
        # pixels in transformed space
        coeffs = self.transform.decompose(pixels)
        # regularization weights
        W = self.weights
        # only first scale (i.e. high frequencies)
        l1_weighted_coeffs0 = jnp.sum(jnp.abs(W[0] * coeffs[0]))
        log_prob = - lambda_0 * l1_weighted_coeffs0
        return log_prob
