# Describes a light model, as a list of light profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

from herculens.LightModel.Profiles import (sersic, pixelated, uniform, 
                                           gaussian, shapelets)
from herculens.Util import util

__all__ = ['LightModelBase']


SUPPORTED_MODELS = [
    'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 
    'SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC',
    'SHAPELETS', 'UNIFORM', 'PIXELATED'
]


class LightModelBase(object):
    """Base class for source and lens light models."""
    def __init__(self, light_model_list, smoothing=0.001,
                 pixel_interpol='bilinear', kwargs_pixelated=None, 
                 shapelets_n_max=4, pixel_allow_extrapolation=False):
        """Create a LightModelBase object.

        Parameters
        ----------
        light_model_list : list of str
            Light model types.
        smoothing : float
            Smoothing factor for some models (deprecated).
        pixel_interpol : string
            Type of interpolation for 'PIXELATED' profiles: 'bilinear' or 'bicubic'
        pixel_allow_extrapolation : bool
            For 'PIXELATED' profiles, wether or not to extrapolate flux values outside the chosen region
            otherwise force values to be zero.
        kwargs_pixelated : dict
            Settings related to the creation of the pixelated grid. See herculens.PixelGrid.create_model_grid for details 

        """
        func_list = []
        pix_idx = None
        for idx, profile_type in enumerate(light_model_list):
            if profile_type == 'GAUSSIAN':
                func_list.append(gaussian.Gaussian())
            elif profile_type == 'GAUSSIAN_ELLIPSE':
                func_list.append(gaussian.GaussianEllipse())
            elif profile_type == 'SERSIC':
                func_list.append(sersic.Sersic(smoothing))
            elif profile_type == 'SERSIC_ELLIPSE':
                func_list.append(sersic.SersicElliptic(smoothing))
            elif profile_type == 'CORE_SERSIC':
                func_list.append(sersic.CoreSersic(smoothing))
            elif profile_type == 'UNIFORM':
                func_list.append(uniform.Uniform())
            elif profile_type == 'PIXELATED':
                if pix_idx is not None:
                    raise ValueError("Multiple pixelated profiles is currently not supported.")
                func_list.append(pixelated.Pixelated(interpolation_type=pixel_interpol, 
                                                     allow_extrapolation=pixel_allow_extrapolation))
                pix_idx = idx
            elif profile_type == 'SHAPELETS':
                func_list.append(shapelets.Shapelets(shapelets_n_max))
            else:
                err_msg = (f"No light model of type {profile_type} found. " +
                           f"Supported types are: {SUPPORTED_MODELS}")
                raise ValueError(err_msg)
        self.func_list = func_list
        self._num_func = len(self.func_list)
        self._pix_idx = pix_idx
        if kwargs_pixelated is None:
            kwargs_pixelated = {}
        self._kwargs_pixelated = kwargs_pixelated

    @property
    def param_name_list(self):
        """Get parameter names as a list of strings for each light model."""
        return [func.param_names for func in self.func_list]

    def _bool_list(self, k):
        return util.convert_bool_list(n=self._num_func, k=k)

    @property
    def has_pixels(self):
        return self._pix_idx is not None

    @property
    def pixel_grid_settings(self):
        return self._kwargs_pixelated

    def set_pixel_grid(self, pixel_grid, data_pixel_area):
        self.func_list[self.pixelated_index].set_pixel_grid(pixel_grid, data_pixel_area)

    @property
    def pixel_grid(self):
        if not self.has_pixels:
            return None
        return self.func_list[self.pixelated_index].pixel_grid

    @property
    def pixelated_index(self):
        # TODO: support multiple pixelated profiles
        return self._pix_idx

    @property
    def pixelated_coordinates(self):
        if not self.has_pixels:
            return None, None
        return self.pixel_grid.pixel_coordinates

    @property
    def pixelated_shape(self):
        if not self.has_pixels:
            return None
        x_coords, _ = self.pixelated_coordinates
        return x_coords.shape

    @property
    def num_amplitudes_list(self):
        return [func.num_amplitudes for func in self.func_list]


