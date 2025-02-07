# Describes a light model, as a list of light profiles
# 
# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp

import herculens.LightModel.profile_mapping as pm
from herculens.Util import util

__all__ = ['LightModelBase']


SUPPORTED_MODELS = pm.SUPPORTED_MODELS
STRING_MAPPING = pm.STRING_MAPPING


class LightModelBase(object):
    """Base class for source and lens light models."""
    # TODO: instead of settings for creating PixelGrid objects, pass directly the object to the LightModel
    def __init__(self, profile_list, kwargs_pixelated=None,
                 **profile_specific_kwargs):
        """Create a LightModelBase object.

        NOTE: the extra keyword arguments are given to the corresponding profile class
        only when that profile is given as a string instead of a class instance.

        Parameters
        ----------
        profile_list : list of strings or profile instances
            List of mass profiles. If not a list, wrap the passed argument in a list. 
        kwargs_pixelated : dict
            Settings related to the creation of the pixelated grid.
            See herculens.PixelGrid.create_model_grid for details.
        profile_specific_kwargs : dict
            See docstring for get_class_from_string().

        """
        if not isinstance(profile_list, (list, tuple)):
            raise TypeError("The profile list should be a list or a tuple.")
        self.func_list, self._pix_idx = self._load_model_instances(
            profile_list, **profile_specific_kwargs
        )
        self._num_func = len(self.func_list)
        self._model_list = profile_list
        if kwargs_pixelated is None:
            kwargs_pixelated = {}
        self._kwargs_pixelated = kwargs_pixelated
        
    def _load_model_instances(
            self, profile_list, **profile_specific_kwargs,
        ):
        func_list = []
        pix_idx = None
        for idx, profile_type in enumerate(profile_list):
            if isinstance(profile_type, str):
                # passing string is supported for backward-compatibility only
                profile_class = self.get_class_from_string(
                    profile_type, 
                    **profile_specific_kwargs,
                )
                if profile_type in ['PIXELATED']:
                    pix_idx = idx

            # this is the new preferred way: passing the profile as a class
            elif self.is_light_profile_class(profile_type):
                profile_class = profile_type
                if isinstance(profile_class, STRING_MAPPING['PIXELATED']):
                    pix_idx = idx
            else:
                raise ValueError("Each profile can either be a string or "
                                 "directly the profile instance.")
            func_list.append(profile_class)
        return func_list, pix_idx
    
    @staticmethod
    def is_light_profile_class(profile):
        """Simply checks that the mass profile has the required methods"""
        return hasattr(profile, 'function')

    @staticmethod
    def get_class_from_string(
            profile_string, 
            smoothing=0.001, 
            shapelets_n_max=4,
            superellipse_exponent=2, 
            pixel_interpol='bilinear', 
            pixel_adaptive_grid=False, 
            pixel_allow_extrapolation=False,
        ):
        """
        Get the profile class of the corresponding type.
        Keyword arguments are related to specific profile types.
            
        Parameters
        ----------
        smoothing : float
            Smoothing factor for some models.
        shapelets_n_max : int
            Maximal order of the shapelets basis set.
        superellipse_exponent : int, float
            Exponent for super-elliptical profiles (e.g. 'SERSIC_SUPERELLIPSE').
        pixel_interpol : string
            Type of interpolation for 'PIXELATED' profiles: 'bilinear' or 'bicubic'
        pixel_adaptive_grid : bool
            Whether or not the pixelated light profile is defined on a grid
            whose extent is adapted based on other model components.
        pixel_allow_extrapolation : bool
            Wether or not to allow the interpolator to predict values outside 
            the field of view of the pixelated profile
        """
        if profile_string in SUPPORTED_MODELS:
            profile_class = STRING_MAPPING[profile_string]
            # treats the few special cases that require user settings
            if profile_string == 'SERSIC':
                return profile_class(smoothing=smoothing)
            elif profile_string == 'SERSIC_ELLIPSE':
                return profile_class(smoothing=smoothing, exponent=2)
            elif profile_string == 'SERSIC_SUPERELLIPSE':
                return profile_class(smoothing=smoothing, exponent=superellipse_exponent)
            elif profile_string == 'SHAPELETS':
                return profile_class(shapelets_n_max)
            elif profile_string == 'PIXELATED':
                return profile_class(interpolation_type=pixel_interpol, 
                                    allow_extrapolation=pixel_allow_extrapolation, 
                                    adaptive_grid=pixel_adaptive_grid)
        else:
            raise ValueError(f"Could not load profile type '{profile_string}'.")
        # all remaining profiles take no extra arguments
        return profile_class()

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
    def pixel_is_adaptive(self):
        if not self.has_pixels:
            return False
        return self.func_list[self.pixelated_index].is_adaptive

    @property
    def num_amplitudes_list(self):
        return [func.num_amplitudes for func in self.func_list]

