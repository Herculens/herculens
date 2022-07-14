# Defines regularization choices
# 
# Copyright (c) 2022, herculens developers and contributors

__author__ = 'aymgal'


import numpy as np
from herculens.RegulModel.Methods import wavelets, positivity
from herculens.Util import util


__all__ = ['RegularizationModel']


SUPPORTED_MODELS = [
    'L1_STARLETS_SOURCE',
    'L1_BLWAVELET_SOURCE',
    'POSITIVITY_SOURCE',
    'L1_STARLETS_POTENTIAL',
    'L1_BLWAVELET_POTENTIAL',
    'POSITIVITY_POTENTIAL',
]


class RegularizationModel(object):

    def __init__(self, regularization_model_list,
                 potential_noise_map=None,
                 potential_mask=None):
        self.method_list = regularization_model_list
        func_list = []
        param_names = []
        for method_type in self.method_list:
            if method_type == 'L1_STARLETS_SOURCE':
                func = wavelets.L1StarletsSource()
            elif method_type == 'L1_BLWAVELET_SOURCE':
                func = wavelets.L1BattleLemarieWaveletSource()
            elif method_type == 'POSITIVITY_SOURCE':
                func = positivity.PositivitySource()
            elif method_type == 'L1_STARLETS_POTENTIAL':
                if potential_noise_map is None:
                    raise ValueError(f"A pre-computed potential noise map must be "
                                     f"provided for regularization '{method_type}'.")
                func = wavelets.L1StarletsPotential(potential_noise_map,
                                                    mask=potential_mask)
            elif method_type == 'L1_BLWAVELET_POTENTIAL':
                if potential_noise_map is None:
                    raise ValueError(f"A pre-computed potential noise map must be "
                                     f"provided for regularization '{method_type}'.")
                func = wavelets.L1BattleLemarieWaveletPotential(potential_noise_map,
                                                                mask=potential_mask)
            elif method_type == 'POSITIVITY_POTENTIAL':
                func = positivity.PositivityPotential()
            else:
                err_msg = (f"No regularization method of type {method} found. " +
                           f"Supported methods are: {SUPPORTED_MODELS}")
                raise ValueError(err_msg)
            func_list.append(func)
            param_names.append(func.param_names)
        self.func_list = func_list
        self.param_names = param_names
        self._num_func = len(self.method_list)

    def initialize_with_lens_image(self, lens_image):
        self._pix_src_idx = lens_image.SourceModel.pixelated_index
        self._pix_lens_idx = lens_image.LensModel.pixelated_index
        self._image_mask = lens_image.image_mask
        for func in self.func_list:
            pixels = func.initialize_with_lens_image(lens_image)

    def log_regularization(self, image_noise_map, kwargs, k=None):
        """Total log-regularization to be added to the loss function.

        Parameters
        ----------
        kwargs : list
            List of parameter dictionaries corresponding to each source model.
        k : int, optional
            Position index of a single source model component.

        """
        logR = 0.
        bool_list = util.convert_bool_list(self._num_func, k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                args = self._get_args(i, image_noise_map, kwargs)
                logR += func(*args, **kwargs['kwargs_regul'][i])
        return logR

    def _get_args(self, i, image_noise_map, kwargs):
        if self.method_list[i] in ['L1_STARLETS_SOURCE', 'L1_BLWAVELET_SOURCE']:
            values = kwargs['kwargs_source'][self._pix_src_idx]['pixels']
            noise = image_noise_map
            mask = self._image_mask
            return (values, noise, mask)
        elif self.method_list[i] in ['POSITIVITY_SOURCE']:
            values = kwargs['kwargs_source'][self._pix_src_idx]['pixels']
            return (values,)
        elif self.method_list[i] in ['L1_STARLETS_POTENTIAL', 'L1_BLWAVELET_POTENTIAL']:
            values = kwargs['kwargs_lens'][self._pix_lens_idx]['pixels']
            noise = None
            mask = None
            return (values, noise, mask)
        elif self.method_list[i] in ['POSITIVITY_POTENTIAL']:
            values = kwargs['kwargs_lens'][self._pix_lens_idx]['pixels']
            return (values,)
        else:
            raise ValueError(f"Regularization method '{self.method_list[i]}' "
                             f"cannot be linked to any mass/light profile.")
