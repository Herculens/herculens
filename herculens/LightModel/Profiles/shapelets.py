# Defines a pixelated profile
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'

import jax.numpy as jnp


__all__= ['Shapelets']


class Shapelets(object):
    """Surface brightness with a set of basis functions"""
    param_names = ['beta', 'center_x', 'center_y', 'amps']
    lower_limit_default = {'beta': 0.,  'center_x': 100., 'center_y': -100., 'amps': -1e10}
    upper_limit_default = {'beta': 1e5, 'center_x': 100., 'center_y':  100., 'amps':  1e10}
    fixed_default = {key: False for key in param_names}

    def __init__(self, n_max, function_type='gaussian'):
        from gigalens.jax.profiles.light.shapelets import Shapelets as ShapeletsGigaLens
        
        if function_type == 'gaussian':
            self._n_max = n_max
            self._backend = ShapeletsGigaLens(self._n_max, use_lstsq=False, interpolate=False)
        else:
            raise ValueError(f"Basis function type '{function_type}' is not supported")
        self._func_type = function_type

    @property
    def maximum_order(self):
        return self._n_max

    @property
    def num_amplitudes(self):
        if self._func_type == 'gaussian':
            return int((self._n_max+1) * (self._n_max+2) / 2)

    def function(self, x, y, beta, center_x, center_y, amps):
        # convert amplitudes to GigaLens syntax
        decimal_places = len(str(self._backend.depth))
        amps_dict = {f'amp{str(i).zfill(decimal_places)}': jnp.array([amps[i]]) for i in range(self.num_amplitudes)}
        return self._backend.light(x, y, center_x, center_y, beta, **amps_dict)
