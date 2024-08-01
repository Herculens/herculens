# Describes a collection of light models used for multi-plane lensing
# 
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'krawczyk'

import jax
import jax.numpy as jnp

from functools import partial
from herculens.LightModel.light_model import LightModel


class MPLightModel(object):
    @staticmethod
    def partial_flux(light_model, x, y, kwargs, k):
        return light_model.surface_brightness(x, y, kwargs, k)

    @staticmethod
    def partial_spatial_derivative(light_model, x, y, kwargs, k):
        return light_model.spatial_derivatives(x, y, kwargs, k)

    def __init__(self, mp_light_model_list, light_model_kwargs):
        '''
        Create a MPLightModel object.

        Parameters
        ----------
        mp_light_model_list : list of str
            List of lists containing light model profiles for each plane of
            the lens system. One inner list per plane with the outer list
            sorted by distance from observer starting with the lens light.
        light_model_kwargs : dictionary for settings related to PIXELATED
            profiles.
        '''
        self.mp_profile_type_list = mp_light_model_list
        self.light_models = []
        self.number_light_planes = len(self.mp_profile_type_list)
        for ldx, light_plane in enumerate(self.mp_profile_type_list):
            self.light_models.append(LightModel(
                light_plane,
                **light_model_kwargs[ldx]
            ))

    @property
    def has_pixels(self):
        return [light_model.has_pixels for light_model in self.light_models]

    @property
    def pixel_is_adaptive(self):
        return [light_model.pixel_is_adaptive for light_model in self.light_models]

    @property
    def pixelated_shape(self):
        return [light_model.pixelated_shape for light_model in self.light_models]

    @property
    def pixelated_coordinates(self):
        return [light_model.pixelated_coordinates for light_model in self.light_models]

    @property
    def pixelated_index(self):
        return [light_model.pixelated_index for light_model in self.light_models]

    @property
    def pixel_grid(self):
        return [light_model.pixel_grid for light_model in self.light_models]

    @property
    def pixel_grid_settings(self):
        return [light_model.pixel_grid_settings for light_model in self.light_models]

    @property
    def param_name_list(self):
        return [light_model.param_name_list for light_model in self.light_models]

    def k_expand(self, k):
        # helper function to expand `None` into a list of `None` values of the
        # expected length.
        if k is None:
            return [None] * self.number_light_planes
        else:
            return k

    @partial(jax.jit, static_argnums=(0, 6))
    def surface_brightness(
        self,
        x,
        y,
        kwargs_list,
        pixels_x_coord,
        pixels_y_coord,
        k=None,
    ):
        '''Total source flux at a given position.

        Parameters
        ----------
        x, y : array_like
            Position coordinate(s) in arcsec relative to the image center for each light
            plane (first index corresponds to light plane).
        kwargs_list : list
            List of lists of parameter dictionaries corresponding to each source model.
        k : list, optional
            List of position index of a single source model component for each light
            plane.
        '''
        k = self.k_expand(k)
        return jnp.stack([
            self.light_models[j].surface_brightness(
                x[j], y[j], kwargs_list[j],
                k=k[j],
                pixels_x_coord=pixels_x_coord[j],
                pixels_y_coord=pixels_y_coord[j]
            )
            for j in range(self.number_light_planes)
        ])

    def spatial_derivatives(self, x, y, kwargs_list, k=None):
        '''Spatial derivatives of the source flux at a given position (along x and y directions).

        Parameters
        ----------
        x, y : array_like
            Position coordinate(s) in arcsec relative to the image center for each light
            plane (first index corresponds to light plane).
        kwargs_list : list
            List of lists of parameter dictionaries corresponding to each source model.
        k : list, optional
            List of position index of a single source model component for each light
            plane.
        '''
        k = self.k_expand(k)
        results = []
        for j in range(self.number_light_planes):
            results.append(self.light_models[j].spatial_derivatives(
                x[j], y[j], kwargs_list[j], k[j]
            ))
        return jnp.stack(results)

