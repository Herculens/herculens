# Defines the model of a strong lens
#
# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'

import copy
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax import random

from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
# from herculens.LensImage.Numerics.numerics import Numerics
from herculens.LensImage.lensing_operator import LensingOperator


__all__ = [
    'LensImage', 
    'LensImage3D',  # TODO: to implement
]


class LensImage(MPLensImage):
    """Generate lensed images from source light, lens mass/light, and point source models."""

    def __init__(self, grid_class, psf_class,
                 noise_class=None, lens_mass_model_class=None,
                 source_model_class=None, lens_light_model_class=None,
                 point_source_model_class=None, 
                 source_arc_mask=None,
                 kwargs_numerics=None,
                 kwargs_lens_equation_solver=None):
        """
        WIP
        """
        if point_source_model_class is not None:
            raise NotImplementedError
        mp_mass_model_list = [lens_mass_model_class.func_list]  # single mass plane
        mp_mass_model_class = MPMassModel(mp_mass_model_list)
        mp_light_model_list = []
        light_model_kwargs = []
        if lens_light_model_class is not None:
            # first light plane
            mp_light_model_list.append(lens_light_model_class.func_list)
            light_model_kwargs.append({'kwargs_pixelated': lens_light_model_class.pixel_grid_settings})
        else:
            mp_light_model_list.append([])
            light_model_kwargs.append({})
        if source_model_class is not None:
            # second light plane
            mp_light_model_list.append(source_model_class.func_list)
            light_model_kwargs.append({'kwargs_pixelated': source_model_class.pixel_grid_settings})
        else:
            mp_light_model_list.append([])
            light_model_kwargs.append({})
        mp_light_model_class = MPLightModel(mp_light_model_list, light_model_kwargs)
        super().__init__(
            grid_class,
            psf_class,
            noise_class,
            mp_mass_model_class,
            mp_light_model_class,
            source_arc_masks=[None, source_arc_mask],  # no mask for lens light plane
            source_grid_scale=None, # TODO
            conjugate_points=None,  # TODO
            kwargs_numerics=kwargs_numerics
        )
        self._eta_flat_fixed = np.array([1.])  # this is fixed in single lens plane mode

    @partial(jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    def model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
              kwargs_point_source=None, unconvolved=False, supersampled=False,
              source_add=True, lens_light_add=True, point_source_add=True,
              k_lens=None, k_source=None, k_lens_light=None, k_point_source=None):
        """
        Create the 2D model image from parameter values.
        Note: due to JIT compilation, the first call to this method will be slower.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_point_source: keyword arguments corresponding to "other" parameters, such as external shear and
                                    point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param supersampled: if True, returns the model on the higher resolution grid (WARNING: no convolution nor normalization is performed in this case!)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, compute point source multiple images, otherwise without
        :param k_lens: list of bool or list of int to select which lens mass profiles to include
        :param k_source: list of bool or list of int to select which source profiles to include
        :param k_lens_light: list of bool or list of int to select which lens light profiles to include
        :param k_point_source: list of bool or list of int to select which point sources to include
        :return: 2d array of surface brightness pixels of the simulation
        """
        if kwargs_point_source is not None:
            raise NotImplementedError
        if unconvolved is True or source_add is False or lens_light_add is False:
            raise NotImplementedError
        if k_lens is None:
            k_mass = None
        else:
            raise NotImplementedError
        if k_lens_light is None and k_source is None:
            k_light = None
        else:
            raise NotImplementedError
        model = super().model(
            eta_flat=self._eta_flat_fixed,
            kwargs_mass=[kwargs_lens],
            kwargs_light=[kwargs_lens_light] + [kwargs_source],
            supersampled=supersampled,
            k_mass=k_mass,
            k_light=k_light,
            k_planes=None,
            return_pixel_scale=False,
        )
        return model

    # def source_surface_brightness(self, kwargs_source, kwargs_lens=None,
    #                               unconvolved=False, supersampled=False,
    #                               de_lensed=False, k=None, k_lens=None):
    #     """

    #     computes the source surface brightness distribution

    #     :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
    #     :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles.
    #     When using an adaptive source pixel grid, kwargs_lens is required even if de_lensed=True.
    #     :param kwargs_extinction: list of keyword arguments of extinction model
    #     :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
    #     :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
    #     :param k: list of bool or list of int to select which source profiles to include
    #     :param k_lens: list of bool or list of int to select which lens mass profiles to include
    #     :return: 2d array of surface brightness pixels
    #     """
    #     if len(self.SourceModel.profile_type_list) == 0:
    #         return jnp.zeros(self.Grid.num_pixel_axes)
    #     x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
    #     source_light = self.eval_source_surface_brightness(
    #         x_grid_img, y_grid_img,
    #         kwargs_source, kwargs_lens=kwargs_lens,
    #         k=k, k_lens=k_lens, de_lensed=de_lensed,
    #     )
    #     if not supersampled:
    #         source_light = self.ImageNumerics.re_size_convolve(
    #             source_light, unconvolved=unconvolved)
    #     return source_light
    
    # def eval_source_surface_brightness(self, x, y, kwargs_source, kwargs_lens=None, 
    #                                    k=None, k_lens=None, de_lensed=False):
    #     if self._src_adaptive_grid:
    #         pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(kwargs_lens, k_lens=k_lens)
    #     else:
    #         pixels_x_coord, pixels_y_coord = None, None  # fall back on fixed, user-defined coordinates
    #     if de_lensed is True:
    #         source_light = self.SourceModel.surface_brightness(x, y, kwargs_source, k=k,
    #                                                            pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord)
    #     else:
    #         x_grid_src, y_grid_src = self.MassModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
    #         source_light = self.SourceModel.surface_brightness(x_grid_src, y_grid_src, kwargs_source, k=k,
    #                                                            pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord)
    #     return source_light

    # def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False,
    #                             supersampled=False, k=None):
    #     """

    #     computes the lens surface brightness distribution

    #     :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
    #     :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
    #     :param k: list of bool or list of int to select which model profiles to include
    #     :return: 2d array of surface brightness pixels
    #     """
    #     x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
    #     lens_light = self.LensLightModel.surface_brightness(x_grid_img, y_grid_img, kwargs_lens_light, k=k)
    #     if not supersampled:
    #         lens_light = self.ImageNumerics.re_size_convolve(
    #             lens_light, unconvolved=unconvolved)
    #     return lens_light

    # def point_source_image(self, kwargs_point_source, kwargs_lens,
    #                        kwargs_solver, k=None):
    #     """Compute PSF-convolved point sources rendered on the image plane.

    #     :param kwargs_point_source: list of keyword arguments corresponding to the point sources
    #     :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
    #     :param k: list of bool or list of int to select which point sources to include
    #     :return: 2d array at the image plane resolution of all multiple images of the point sources
    #     """
    #     result = jnp.zeros((self.Grid.num_pixel_axes))
    #     if self.PointSourceModel is None:
    #         return result
    #     theta_x, theta_y, amplitude = self.PointSourceModel.get_multiple_images(
    #         kwargs_point_source, kwargs_lens=kwargs_lens, kwargs_solver=kwargs_solver, 
    #         k=k, with_amplitude=True, zero_amp_duplicates=True
    #     )
    #     for i in range(len(theta_x)):
    #         result += self.ImageNumerics.render_point_sources(
    #             theta_x[i], theta_y[i], amplitude[i]
    #         )
    #     return result

