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
    def __init__(self, profile_list, 
                 kwargs_pixelated=None, 
                 no_complex_numbers=True,
                 pixel_interpol='fast_bilinear', 
                 pixel_derivative_type='interpol',
                 kwargs_pixel_grid_fixed=None):
        """Create a MassProfileBase object.

        NOTE: extra keyword arguments are given to the corresponding profile class
        only when that profile is given as a string instead of a class instance.

        Parameters
        ----------
        profile_list : list of str or profile class instance
            Lens model profile types.

        """
        self.func_list, self._pix_idx = self._load_model_instances(
            profile_list, pixel_derivative_type, pixel_interpol, 
            no_complex_numbers, kwargs_pixel_grid_fixed
        )
        self._num_func = len(self.func_list)
        self._model_list = profile_list
        if kwargs_pixelated is None:
            kwargs_pixelated = {}
        self._kwargs_pixelated = kwargs_pixelated
        
    def _load_model_instances(
            self, profile_list, pixel_derivative_type, pixel_interpol, 
            no_complex_numbers, kwargs_pixel_grid_fixed,
        ):
        func_list = []
        pix_idx = None
        for idx, profile_type in enumerate(profile_list):
            # NOTE: Passing string is supported for backward-compatibility only
            if isinstance(profile_type, str):
                # These models require a new instance per profile as certain pre-computations
                # are relevant per individual profile
                profile_class = self.get_class_from_string(
                    profile_type, 
                    kwargs_pixel_grid_fixed=kwargs_pixel_grid_fixed,
                    pixel_derivative_type=pixel_derivative_type, 
                    pixel_interpol=pixel_interpol,
                    no_complex_numbers=no_complex_numbers, 
                )
                if profile_type in ['PIXELATED', 'PIXELATED_DIRAC']:
                    pix_idx = idx

            # NOTE: this is the new preferred way: passing the profile as a class
            elif self.is_mass_profile_class(profile_type):
                profile_class = profile_type
                if isinstance(
                        profile_class, 
                        (STRING_MAPPING['PIXELATED'], STRING_MAPPING['PIXELATED_DIRAC'])
                    ):
                    pix_idx = idx
            else:
                raise ValueError("Each profile can either be a string or "
                                 "directly the profile class (not instantiated).")
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
            profile_string, pixel_derivative_type=None, pixel_interpol=None, 
            no_complex_numbers=None, kwargs_pixel_grid_fixed=None,
        ):
        """Get the lens profile class of the corresponding type."""
        if profile_string not in list(STRING_MAPPING.keys()):
            raise ValueError(f"{profile_string} is not a valid lens model. "
                             f"Supported types are {SUPPORTED_MODELS}")
        profile_class = STRING_MAPPING[profile_string]
        # treats the few special cases that require user settings
        if profile_string == 'EPL':
            return profile_class(no_complex_numbers=no_complex_numbers)
        elif profile_string == 'PIXELATED':
            return profile_class(derivative_type=pixel_derivative_type, interpolation_type=pixel_interpol)
        elif profile_string == 'PIXELATED_FIXED':
            if kwargs_pixel_grid_fixed is None:
                raise ValueError("At least one pixel grid must be provided to use 'PIXELATED_FIXED' profile")
            return profile_class(**kwargs_pixel_grid_fixed)
        # all remaining profile takes no extra arguments
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
