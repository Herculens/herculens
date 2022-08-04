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
from herculens.Util.util import convert_bool_list

__all__ = ['LightModelBase']


SUPPORTED_MODELS = [
    'GAUSSIAN', 'GAUSSIAN_ELLIPSE', 
    'SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC',
    'SHAPELETS', 'UNIFORM', 'PIXELATED']


class LightModelBase(object):
    """Base class for source and lens light models."""
    def __init__(self, light_model_list, smoothing=0.001,
                 pixel_interpol='bilinear', pixel_allow_extrapolation=False,
                 kwargs_pixelated={}, shapelets_n_max=4):
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
        self.profile_type_list = light_model_list
        func_list = []
        for profile_type in light_model_list:
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
                func_list.append(pixelated.Pixelated(method=pixel_interpol, allow_extrapolation=pixel_allow_extrapolation))
            elif profile_type == 'SHAPELETS':
                func_list.append(shapelets.Shapelets(shapelets_n_max))
            else:
                err_msg = (f"No light model of type {profile_type} found. " +
                           f"Supported types are: {SUPPORTED_MODELS}")
                raise ValueError(err_msg)
        self.func_list = func_list
        self._num_func = len(self.func_list)
        self._kwargs_pixelated = kwargs_pixelated

    @property
    def param_name_list(self):
        """Get parameter names as a list of strings for each light model."""
        return [func.param_names for func in self.func_list]

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """Total source flux at a given position.

        Parameters
        ----------
        x, y : float or array_like
            Position coordinate(s) in arcsec relative to the image center.
        kwargs_list : list
            List of parameter dictionaries corresponding to each source model.
        k : int, optional
            Position index of a single source model component.

        """
        # x = jnp.array(x, dtype=float)
        # y = jnp.array(y, dtype=float)
        flux = 0.
        bool_list = convert_bool_list(self._num_func, k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                flux += func.function(x, y, **kwargs_list[i])
        return flux

    def spatial_derivatives(self, x, y, kwargs_list, k=None):
        """Spatial derivatives of the source flux at a given position (along x and y directions).

        Parameters
        ----------
        x, y : float or array_like
            Position coordinate(s) in arcsec relative to the image center.
        kwargs_list : list
            List of parameter dictionaries corresponding to each source model.
        k : int, optional
            Position index of a single source model component.

        """
        x = jnp.array(x, dtype=float)
        y = jnp.array(y, dtype=float)
        # flux = jnp.zeros_like(x)
        f_x, f_y = 0., 0.
        bool_list = convert_bool_list(self._num_func, k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                f_x_, f_y_ = func.derivatives(x, y, **kwargs_list[i])
                f_x += f_x_
                f_y += f_y_
        return f_x, f_y

    @property
    def has_pixels(self):
        return ('PIXELATED' in self.profile_type_list)

    @property
    def pixel_grid_settings(self):
        return self._kwargs_pixelated

    def set_pixel_grid(self, pixel_axes, data_pixel_area):
        for i, func in enumerate(self.func_list):
            if self.profile_type_list[i] == 'PIXELATED':
                func.set_data_pixel_grid(pixel_axes, data_pixel_area)

    @property
    def pixelated_index(self):
        # TODO: what if there are more than one PIXELATED profiles?
        if not hasattr(self, '_pix_idx'):
            self._pix_idx = None
            if self.has_pixels:
                self._pix_idx = self.profile_type_list.index('PIXELATED')
        return self._pix_idx

    @property
    def pixelated_coordinates(self):
        if not self.has_pixels:
            return None, None
        return (self.func_list[self.pixelated_index].x_coords, 
                self.func_list[self.pixelated_index].y_coords)

    @property
    def pixelated_shape(self):
        if not self.has_pixels:
            return None
        x_coords, y_coords = self.pixelated_coordinates
        return (len(y_coords), len(x_coords))

    @property
    def num_amplitudes_list(self):
        return [func.num_amplitudes for func in self.func_list]


