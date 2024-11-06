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
    # 'LensImage3D',  # TODO: to implement
]


class LensImage(MPLensImage):
    """Generate lensed images from source light, lens mass/light, and point source models."""

    def __init__(self, grid_class, psf_class,
                 noise_class=None, lens_mass_model_class=None,
                 source_model_class=None, lens_light_model_class=None,
                 point_source_model_class=None, 
                 source_arc_mask=None, source_grid_scale=None,
                 kwargs_numerics=None,
                 kwargs_lens_equation_solver=None):
        """
        WIP
        """
        # mp_mass_model_list = [lens_mass_model_class.func_list]  # single mass plane
        mp_mass_model_class = MPMassModel([lens_mass_model_class])  # TODO: do the same with LightModels below (i.e. give directly the class, not the strings)
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
            source_grid_scale=[None, source_grid_scale],  # no grid scale for lens light
            conjugate_points=None,  # TODO
            kwargs_numerics=kwargs_numerics
        )
        self._eta_flat_fixed = np.array([1.])  # this is fixed in single lens plane mode

        # the following attributes are to mimick the original implementation of LensImage
        self.MassModel = lens_mass_model_class
        self.LensLightModel = lens_light_model_class
        self.SourceModel = source_model_class  # in single plane there is 1 source model
        if point_source_model_class is not None:
            raise NotImplementedError("Point source are not implemented in this class yet")
        else:
            from herculens.PointSourceModel.point_source_model import PointSourceModel
            point_source_model_class = PointSourceModel(
                point_source_type_list=[], mass_model=lens_mass_model_class,
            )
        self.PointSourceModel = point_source_model_class

    # WIP
    def set_static_model_grid(self):
        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
        for i in range(self.MPMassModel.number_mass_planes):
            for j in range(self.MPMassModel.mass_models[i]._num_func):
                profile = self.MPMassModel.mass_models[i].func_list[j]
                if hasattr(profile, 'set_eval_coord_grid'):
                    profile.set_eval_coord_grid(ra_grid_img, dec_grid_img)
                    print(f"Set static coordinate grid for profile {j} in mass plane {i}")
            

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

    # def adapt_source_coordinates(self, kwargs_lens, k_lens=None, return_plt_extent=False):
    #     # NOTE: despite the keyword name, this does *NOT* return the extent
    #     # as expected from pyplot routines! # TODO: fix this
    #     if k_lens is not None:
    #         raise NotImplementedError("k_lens != None not implemented in new LensImage class.")
    #     ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
    #     kwargs_mass = [kwargs_lens]  # wraps the single mass model for multiplane conventions
    #     ra_grid_planes, dec_grid_planes = self.MPMassModel.ray_shooting(
    #         ra_grid_img,
    #         dec_grid_img,
    #         eta_flat,
    #         kwargs_mass
    #     )
    #     x_coord_list, y_coord_list, extent_list = super().adapt_source_coordinates(
    #         ra_grid_planes, 
    #         dec_grid_planes,
    #         force=False,
    #         npix_src=100,  # not used when force=False
    #         source_grid_scale=1.0  # TODO: implement this
    #     )
    #     # extract the source as the last light model in multiplane conventions
    #     x_coord, y_coord, extent = x_coord_list[-1], y_coord_list[-1], extent_list[-1]
    #     return (x_coord, y_coord, extent)

    def get_source_coordinates(self, kwargs_lens, k_lens=None, return_plt_extent=False):
        """
        WARNING: this does not return the same array shape as the super() class!
        This function returns 2D 'mesh-gridded' arrays so they can be used by
        the plotting routines, and is consistent with the output of the 
        implementation in the original LensImage class.
        """
        if k_lens is not None:
            raise NotImplementedError("k_lens != None not implemented in new LensImage class.")
        # NOTE: despite the keyword name, this does *NOT* return the extent
        # as expected from pyplot routines! # TODO: fix this 
        kwargs_mass = [kwargs_lens]  # wraps the single mass model for multiplane conventions
        x_coord_list, y_coord_list, extent_list = super().get_source_coordinates(
            self._eta_flat_fixed,
            kwargs_mass,
            force=False,
            npix_src=100,  # not used when force=False
            source_grid_scale=self._source_grid_scale[-1], 
        )
        # extract the source as the last light model in multiplane conventions
        x_coord, y_coord, extent = x_coord_list[-1], y_coord_list[-1], extent_list[-1]
        # generate the 2D coordinates grid
        x_grid, y_grid = np.meshgrid(x_coord, y_coord)
        return (x_grid, y_grid, extent)

    def eval_source_surface_brightness(self, x, y, kwargs_source, kwargs_lens=None, 
                                       k=None, k_lens=None, de_lensed=False):
        # pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
        #     kwargs_lens, k_lens=k_lens,
        # )

        if k_lens is not None:
            raise NotImplementedError("k_lens != None not implemented in new LensImage class.")

        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
        kwargs_mass = [kwargs_lens]  # wraps the single mass model for multiplane conventions
        ra_grid_planes, dec_grid_planes = self.MPMassModel.ray_shooting(
            ra_grid_img,
            dec_grid_img,
            self._eta_flat_fixed,
            kwargs_mass
        )
        x_coord_list, y_coord_list, _ = super().adapt_source_coordinates(
            ra_grid_planes, 
            dec_grid_planes,
            force=False,
            npix_src=100,  # not used when force=False
            source_grid_scale=self._source_grid_scale[-1], 
        )
        # extract the source as the last light model in multiplane conventions
        pixels_x_coord, pixels_y_coord = x_coord_list[-1], y_coord_list[-1]

        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(x, y, kwargs_source, k=k,
                                                               pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord)
        else:
            kwargs_mass = [kwargs_lens]
            x_grid_src, y_grid_src = self.MPMassModel.ray_shooting(x, y, kwargs_mass, k=k_lens)
            source_light = self.SourceModel.surface_brightness(x_grid_src, y_grid_src, kwargs_source, k=k,
                                                               pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord)
        return source_light

    def normalized_residuals(self, data, model, mask=None):
        """
        compute the map of normalized residuals,
        given the data and the model image
        """
        if mask is None:
            mask = np.ones(self.Grid.num_pixel_axes)
        noise_var = self.Noise.C_D_model(model)
        noise = np.sqrt(noise_var)
        norm_res_model = (data - model) / noise * mask
        norm_res_tot = norm_res_model
        if mask is not None:
            # outside the mask just add pure data
            norm_res_tot += (data / noise) * (1. - mask)
        # make sure there is no NaN or infinite values
        norm_res_model = np.where(np.isfinite(norm_res_model), norm_res_model, 0.)
        norm_res_tot = np.where(np.isfinite(norm_res_tot), norm_res_tot, 0.)
        return norm_res_model, norm_res_tot

    def reduced_chi2(self, data, model, mask=None):
        """
        compute the reduced chi2 of the data given the model
        """
        if mask is None:
            mask = np.ones(self.Grid.num_pixel_axes)
        norm_res, _ = self.normalized_residuals(data, model, mask=mask)
        num_data_points = np.sum(mask)
        return np.sum(norm_res**2) / num_data_points
        