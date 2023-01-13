# Defines the model of a strong lens
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the ImSim module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'

import copy
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit

from herculens.LensImage.Numerics.numerics import Numerics


__all__ = ['LensImage']


class LensImage(object):
    """Generate lensed images from source light, lens mass/light, and point source models."""
    def __init__(self, grid_class, psf_class,
                 noise_class=None, lens_mass_model_class=None,
                 source_model_class=None, lens_light_model_class=None,
                 point_source_model_class=None, kwargs_numerics=None):
        """
        :param grid_class: coordinate system, instance of PixelGrid() from herculens.Coordinates.pixel_grid
        :param psf_class: point spread function, instance of PSF() from herculens.Instrument.psf
        :param noise_class: noise properties, instance of Noise() from herculens.Instrument.noise
        :param lens_mass_model_class: lens mass model, instance of MassModel() from herculens.MassModel.mass_model
        :param source_model_class: extended source light model, instance of LightModel() from herculens.MassModel.mass_model
        :param lens_light_model_class: lens light model, instance of LightModel() from herculens.MassModel.mass_model
        :param point_source_model_class: point source model, instance of PointSourceModel() from herculens.PointSource.point_source
        :param kwargs_numerics: keyword arguments for various numerical settings (see herculens.Numerics.numerics)
        """
        self.Grid = grid_class
        self.PSF = psf_class
        self.Noise = noise_class

        # Require now that all relevant parameters of the PSF model are provided
        # when it is instantiated (outside this object). This helps avoid JAX
        # tracer errors due to, e.g., the PSF kernel size being computed for
        # the first time inside the jitted model() method, where it would
        # cause the kernel to become an abstract tracer
        #
        # self.PSF.set_pixel_size(self.Grid.pixel_width)

        if lens_mass_model_class is None:
            from herculens.MassModel.mass_model import MassModel
            lens_mass_model_class = MassModel(mass_model_list=[])
        self.MassModel = lens_mass_model_class
        if self.MassModel.has_pixels:
            pixel_grid = self.Grid.create_model_grid(**self.MassModel.pixel_grid_settings)
            self.MassModel.set_pixel_grid(pixel_grid)

        if source_model_class is None:
            from herculens.LightModel.light_model import LightModel
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class
        if self.SourceModel.has_pixels:
            pixel_grid = self.Grid.create_model_grid(**self.SourceModel.pixel_grid_settings)
            self.SourceModel.set_pixel_grid(pixel_grid, self.Grid.pixel_area)

        if lens_light_model_class is None:
            from herculens.LightModel.light_model import LightModel
            lens_light_model_class = LightModel(light_model_list=[])
        self.LensLightModel = lens_light_model_class
        if self.LensLightModel.has_pixels:
            pixel_grid = self.Grid.create_model_grid(**self.LensLightModel.pixel_grid_settings)
            self.LensLightModel.set_pixel_grid(pixel_grid, self.Grid.pixel_area)

        if point_source_model_class is None:
            from herculens.PointSourceModel.point_source_model import PointSourceModel
            point_source_model_class = PointSourceModel(point_source_type_list=[], mass_model=self.MassModel)
        self.PointSourceModel = point_source_model_class
        # self.PointSource.update_lens_model(lens_model_class=lens_model_class)
        # x_center, y_center = self.Data.center
        # self.PointSource.update_search_window(search_window=np.max(self.Data.width), x_center=x_center,
        #                                       y_center=y_center, min_distance=self.Data.pixel_width,
        #                                       only_from_unspecified=True)

        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.kwargs_numerics = kwargs_numerics
        self.ImageNumerics = Numerics(pixel_grid=self.Grid, psf=self.PSF, **self.kwargs_numerics)

    def source_surface_brightness(self, kwargs_source, kwargs_lens=None,
                                  unconvolved=False, supersampled=False,
                                  de_lensed=False, k=None, k_lens=None):
        """

        computes the source surface brightness distribution

        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_extinction: list of keyword arguments of extinction model
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :param k: list of bool or list of int to select which source profiles to include
        :param k_lens: list of bool or list of int to select which lens mass profiles to include
        :return: 2d array of surface brightness pixels
        """
        if len(self.SourceModel.profile_type_list) == 0:
            return jnp.zeros((self.Grid.num_pixel_axes))
        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(ra_grid_img, dec_grid_img, kwargs_source, k=k)
        else:
            ra_grid_src, dec_grid_src = self.MassModel.ray_shooting(ra_grid_img, dec_grid_img, kwargs_lens, k=k_lens)
            source_light = self.SourceModel.surface_brightness(ra_grid_src, dec_grid_src, kwargs_source, k=k)
        if not supersampled:
            source_light = self.ImageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return source_light

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False,
                                supersampled=False, k=None):
        """

        computes the lens surface brightness distribution

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :param k: list of bool or list of int to select which model profiles to include
        :return: 2d array of surface brightness pixels
        """
        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(ra_grid_img, dec_grid_img, kwargs_lens_light, k=k)
        if not supersampled:
            lens_light = self.ImageNumerics.re_size_convolve(lens_light, unconvolved=unconvolved)
        return lens_light

    def point_source_image(self, kwargs_point_source, kwargs_lens, k=None):
        """Compute PSF-convolved point sources rendered on the image plane.

        :param kwargs_point_source: list of keyword arguments corresponding to the point sources
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param k: list of bool or list of int to select which point sources to include
        :return: 2d array at the image plane resolution of all multiple images of the point sources
        """
        result = jnp.zeros((self.Grid.num_pixel_axes))
        if self.PointSourceModel is None:
            return result

        theta_x, theta_y, amplitude = self.PointSourceModel.get_multiple_images(kwargs_point_source, kwargs_lens, k)
        for i in range(len(theta_x)):
            result += self.ImageNumerics.render_point_sources(theta_x[i], theta_y[i], amplitude[i])

        return result

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
        # TODO: simplify treatment of convolution, downsampling and re-sizing
        model = jnp.zeros((self.Grid.num_pixel_axes))
        if supersampled:
            model = jnp.zeros((self.ImageNumerics.grid_class.num_grid_points,))

        if source_add:
            model += self.source_surface_brightness(kwargs_source, kwargs_lens,
                                                    unconvolved=unconvolved,
                                                    supersampled=supersampled,
                                                    k=k_source, k_lens=k_lens)

        if lens_light_add:
            model += self.lens_surface_brightness(kwargs_lens_light,
                                                  unconvolved=unconvolved,
                                                  supersampled=supersampled,
                                                  k=k_lens_light)

        if point_source_add:
            model += self.point_source_image(kwargs_point_source, kwargs_lens,
                                             k=k_point_source)

        return model

    def simulation(self, add_poisson=True, add_gaussian=True,
                   compute_true_noise_map=True, noise_seed=18,
                   **model_kwargs):
        """
        same as model() but with noise added

        :param compute_true_noise_map: if True (default), define the noise map (diagonal covariance matrix)
        to be the 'true' one, i.e. based on the noiseless model image.
        :param noise_seed: the seed that will be used by the PRNG from JAX to fix the noise realization.
        The default is the arbtrary value 18, so it is the user task to change it for different realizations.
        """
        if self.Noise is None:
            raise ValueError("Impossible to generate noise realisation because no noise class has been set")
        model = self.model(**model_kwargs)
        noise = self.Noise.realisation(model, noise_seed, add_poisson=add_poisson, add_gaussian=add_gaussian)
        simu = model + noise
        self.Noise.set_data(simu)
        if compute_true_noise_map is True:
            self.Noise.compute_noise_map_from_model(model)
        return simu

    def normalized_residuals(self, data, model, mask=None):
        """
        compute the map of normalized residuals,
        given the data and the model image
        """
        if mask is None:
            mask = np.ones(self.Grid.num_pixel_axes)
        noise_var = self.Noise.C_D_model(model)
        # noise_var = self.Noise.C_D
        norm_res = (model - data) / np.sqrt(noise_var) * mask
        return norm_res

    def reduced_chi2(self, data, model, mask=None):
        """
        compute the reduced chi2 of the data given the model
        """
        if mask is None:
            mask = np.ones(self.Grid.num_pixel_axes)
        norm_res = self.normalized_residuals(data, model, mask=mask)
        num_data_points = np.sum(mask)
        return np.sum(norm_res**2) / num_data_points
