# Defines regularization choices
# 
# Copyright (c) 2023, herculens developers and contributors


__author__ = 'aymgal'


import numpy as np
from herculens.RegulModel.Methods import sparsity, constraints
from herculens.Util import util


__all__ = ['RegularizationModel']


SUPPORTED_MODELS = {
    'source': [
        'SPARSITY_STARLET',
        'SPARSITY_STARLET_2',
        'SPARSITY_BLWAVELET',
        'POSITIVITY',
    ],
    'lens_light': [
        'SPARSITY_STARLET',
        'SPARSITY_STARLET_2',
        'SPARSITY_BLWAVELET',
        'POSITIVITY',
    ],
    'lens_mass': [
        'SPARSITY_STARLET_POTENTIAL',
        'SPARSITY_STARLET_2_POTENTIAL',
        'SPARSITY_BLWAVELET_POTENTIAL',
        'POSITIVITY_POTENTIAL',
        'NEGATIVITY_POTENTIAL',
        'POSITIVITY_CONVERGENCE',
    ]
}


class RegularizationModel(object):

    """
    list items should be tuple (model_type, profile_index, method)
    e.g. method_list = [
        ('source', 0, 'SPARSITY_STARLET'),
        ('source', 0, 'POSITIVITY'),
        ('lens_mass', 0, 'SPARSITY_STARLET_2_POTENTIAL'),
    ]
    """

    _model_types = ['source', 'lens_light', 'lens_mass']

    def __init__(self, method_type_list):
        self.method_list, self.param_names = self._setup_methods(method_type_list)

    def initialize(self, lens_image, kwargs_params, **kwargs):
        for method in self.method_list:
            method.initialize(lens_image, kwargs_params, **kwargs)

    def log_prob(self, kwargs_params, kwargs_hyperparams):
        """
        Returns the log-probability term associated to regularization methods,
        given a set of (model) parameters and (regularization) hyperparameters
        """
        log_prob = 0.
        for i, method in enumerate(self.method_list):
            log_prob += method.log_prob(kwargs_params, **kwargs_hyperparams[i])
        return log_prob

    def get_weights(self):
        weights_list = []
        for method in self.method_list:
            if method.has_weights:
                weights_list.append(method.weights)
            else:
                weights_list.append(None)
        return weights_list

    @staticmethod
    def _setup_methods(method_type_list):
        method_list, param_names = [], []
        for model_type, profile_index, method_type in method_type_list:
            if model_type in ('source', 'lens_light'):
                if method_type == 'SPARSITY_STARLET':
                    method = sparsity.SparsityStarlet(model_type, profile_index)
                elif method_type == 'SPARSITY_STARLET_2':
                    method = sparsity.SparsityStarlet(model_type, profile_index,
                                                      second_gen=True)
                elif method_type == 'SPARSITY_BLWAVELET':
                    method = sparsity.SparsityBLWavelet(model_type, profile_index)
                elif method_type == 'POSITIVITY':
                    method = constraints.Positivity(model_type, profile_index)
            elif model_type == 'lens_mass':
                if method_type == 'SPARSITY_STARLET_POTENTIAL':
                    method = sparsity.SparsityStarlet(model_type, profile_index,
                                                     mass_form='potential')
                elif method_type == 'SPARSITY_STARLET_2_POTENTIAL':
                    method = sparsity.SparsityStarlet(model_type, profile_index,
                                                      mass_form='potential',
                                                      second_gen=True)
                elif method_type == 'SPARSITY_BLWAVELET_POTENTIAL':
                    method = sparsity.SparsityBLWavelet(model_type, profile_index,
                                                      mass_form='potential')
                elif method_type == 'POSITIVITY_POTENTIAL':
                    method = constraints.Positivity(model_type, profile_index,
                                                    mass_form='potential')
                elif method_type == 'NEGATIVITY_POTENTIAL':
                    method = constraints.Negativity(model_type, profile_index,
                                                    mass_form='potential')
                elif method_type == 'POSITIVITY_CONVERGENCE':
                    method = constraints.Positivity(model_type, profile_index,
                                                    mass_form='convergence')
            else:
                raise ValueError(f"Regularization method is unsupported "
                                 f"(supported model types are {self._model_types} "
                                 f"and supported methods are {SUPPORTED_MODELS[model_type]})")
            method_list.append(method)
            param_names.append(method.param_names)
        return method_list, param_names
