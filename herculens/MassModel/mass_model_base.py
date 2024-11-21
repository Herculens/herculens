# Describes a mass model, as a list of mass profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import herculens.MassModel.profile_mapping as pm
from herculens.Util import util


__all__ = ['MassModelBase']


SUPPORTED_MODELS = pm.SUPPORTED_MODELS
STRING_MAPPING = pm.STRING_MAPPING


class MassModelBase(object):
    """Base class for managing lens models in single- or multi-plane lensing."""
    def __init__(self, profile_list, kwargs_pixelated=None, 
                 **profile_specific_kwargs):
        """Create a MassProfileBase object.

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
            profile_list, **profile_specific_kwargs,
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
                if profile_type in ['PIXELATED', 'PIXELATED_DIRAC']:
                    pix_idx = idx

            # this is the new preferred way: passing the profile as a class
            elif self.is_mass_profile_class(profile_type):
                profile_class = profile_type
                if isinstance(
                        profile_class, 
                        (STRING_MAPPING['PIXELATED'], STRING_MAPPING['PIXELATED_DIRAC'])
                    ):
                    pix_idx = idx
            else:
                raise TypeError(f"Each profile can either be a string or "
                                 f"directly a profile instance (supported profiles are {SUPPORTED_MODELS}).")
            func_list.append(profile_class)
        return func_list, pix_idx
    
    @staticmethod
    def is_mass_profile_class(profile):
        """Simply checks that the mass profile has the required methods"""
        return (
            hasattr(profile, 'function') and
            hasattr(profile, 'derivatives') and
            hasattr(profile, 'hessian')
        )

    @staticmethod
    def get_class_from_string(
            profile_string, 
            pixel_derivative_type=None, 
            pixel_interpol=None, 
            no_complex_numbers=None, 
            kwargs_pixel_grid_fixed=None,
        ):
        """
        Get the lens profile class of the corresponding type.
        Keyword arguments are related to specific profile types.
        
        Parameters
        ----------
        smoothing : float
            Smoothing factor for some models (deprecated).
        pixel_interpol : string
            Type of interpolation for 'PIXELATED' profiles: 'fast_bilinear' or 'bicubic'
        pixel_derivative_type : str
            Type of interpolation: 'interpol' or 'autodiff'
        no_complex_numbers : bool
            Use or not complex number in the EPL's deflection computation.
        kwargs_pixel_grid_fixed : dict
            Settings related to the creation of the pixelated grid for profile type 'PIXELATED_FIXED'.
            See herculens.PixelGrid.create_model_grid for details.
        """
        if profile_string in SUPPORTED_MODELS:
            profile_class = STRING_MAPPING[profile_string]
            kwargs = {}
            # treats the few special cases that require user settings
            if profile_string == 'EPL':
                if no_complex_numbers is not None:
                    kwargs['no_complex_numbers'] = no_complex_numbers
                return profile_class(**kwargs)
            elif profile_string == 'PIXELATED':
                if pixel_interpol is None:
                    kwargs['interpolation_type'] = pixel_interpol
                if pixel_derivative_type is None:
                    kwargs['derivative_type'] = pixel_derivative_type
                return profile_class(**kwargs)
            elif profile_string == 'PIXELATED_FIXED':
                if kwargs_pixel_grid_fixed is None:
                    raise ValueError("At least one pixel grid must be provided to use 'PIXELATED_FIXED' profile")
                return profile_class(**kwargs_pixel_grid_fixed)
        else:
            raise ValueError(f"Could not load profile type '{profile_string}'.")
        # all remaining profiles take no extra arguments
        return profile_class()

    def _bool_list(self, k):
        return util.convert_bool_list(n=self._num_func, k=k)

    @property
    def has_pixels(self):
        return self._pix_idx is not None

    @property
    def pixel_grid_settings(self):
        return self._kwargs_pixelated

    def set_pixel_grid(self, pixel_grid):
        self.func_list[self.pixelated_index].set_pixel_grid(pixel_grid)

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
