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

    def __init__(self, mp_light_model_list, **light_model_kwargs):
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
        string_input = all([isinstance(lm, str) or lm is None for lm in mp_light_model_list])
        instance_input = all([isinstance(lm, LightModel) or lm is None for lm in mp_light_model_list])
        if string_input:
            self.light_models = [LightModel(lm, **light_model_kwargs) if lm is None else None for lm in mp_light_model_list]
            self.mp_profile_type_list = mp_light_model_list
        elif instance_input:
            self.light_models = mp_light_model_list
            self.mp_profile_type_list = [lm.func_list if lm is not None else None for lm in self.light_models]
        else:
            raise ValueError(
                "MPLightModel needs to be initialized either with a list of lists of strings, "
                "or directly with a list of (single plane) LightModel instances."
            )
        self.number_light_planes = len(self.mp_profile_type_list)

    @property
    def has_pixels(self):
        return [light_model.has_pixels if light_model is not None else False for light_model in self.light_models]
    
    @property
    def has_light(self):
        return [light_model is not None for light_model in self.light_models]

    @property
    def pixel_is_adaptive(self):
        return [light_model.pixel_is_adaptive if light_model is not None else False for light_model in self.light_models]

    @property
    def pixelated_shape(self):
        return [light_model.pixelated_shape if light_model is not None else None for light_model in self.light_models]

    @property
    def pixelated_coordinates(self):
        return [light_model.pixelated_coordinates if light_model is not None else None for light_model in self.light_models]

    @property
    def pixelated_index(self):
        return [light_model.pixelated_index if light_model is not None else None for light_model in self.light_models]

    @property
    def pixel_grid(self):
        return [light_model.pixel_grid if light_model is not None else None for light_model in self.light_models]

    @property
    def pixel_grid_settings(self):
        return [light_model.pixel_grid_settings if light_model is not None else None for light_model in self.light_models]

    @property
    def param_name_list(self):
        return [light_model.param_name_list if light_model is not None else None for light_model in self.light_models]

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
        pixel_x_coords, pixel_y_coords : array_like
            Pixel position coordinate(s) for an adaptive grid based on arc masks (see MPLensImage).
        k : list of list, optional
            only evaluate the k-th light model (list of list of index values) for each light
            plane, by default None

        Returns
        -------
        array_like
            Flux for each image plane in the lens system (first index of each corresponds to
            light plane).
        '''
        k = self.k_expand(k)
        flux = []
        for j in range(self.number_light_planes):
            if self.has_light[j]:
                flux.append(
                    self.light_models[j].surface_brightness(
                        x[j], y[j], kwargs_list[j],
                        k=k[j],
                        pixels_x_coord=pixels_x_coord[j],
                        pixels_y_coord=pixels_y_coord[j]
                    )
                )
        return jnp.stack(flux)

    def spatial_derivatives(self, x, y, kwargs_list, k=None):
        '''Spatial derivatives of the source flux at a given position (along x and y directions)
        for each light plane.

        Parameters
        ----------
        x, y : array_like
            Position coordinate(s) in arcsec relative to the image center for each light
            plane (first index corresponds to light plane).
        kwargs_list : list
            List of lists of parameter dictionaries corresponding to each source model.
        list of list, optional
            only evaluate the k-th light model (list of list of index values) for each light
            plane, by default None

        Returns
        -------
        array_like
            Spatial derivatives of the source flux at a given position for each light plane
            (first index of each corresponds to light plane).
        '''
        k = self.k_expand(k)
        results = []
        for j in range(self.number_light_planes):
            results.append(self.light_models[j].spatial_derivatives(
                x[j], y[j], kwargs_list[j], k[j]
            ))
        return jnp.stack(results)

