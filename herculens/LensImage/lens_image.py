import functools
import copy
import jax.numpy as np
from functools import partial
from jax import jit

from herculens.LensImage.Numerics.numerics_subframe import NumericsSubFrame
from herculens.LensImage.image2source_mapping import Image2SourceMapping
from herculens.Util import util

__all__ = ['ImageModel']


class LensImage(object):
    """Generate lensed images from source light and lens mass/light models."""
    def __init__(self, grid_class, psf_class, 
                 noise_class=None, lens_model_class=None,
                 source_model_class=None, lens_light_model_class=None,
                 kwargs_numerics=None):
        """
        :param grid_class: instance of PixelGrid() class
        :param psf_class: instance of PSF() class
        :param lens_model_class: instance of LensModel() class
        :param source_model_class: instance of LightModel() class describing the source parameters
        :param lens_light_model_class: instance of LightModel() class describing the lens light parameters
        :param point_source_class: instance of PointSource() class describing the point sources
        :param kwargs_numerics: keyword arguments with various numeric description (see ImageNumerics class for options)
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver (see SLITronomy documentation)
        """
        self.type = 'single-band'
        self.num_bands = 1
        self.PSF = psf_class
        self.Noise = noise_class
        # here we deep-copy the class to prevent issues with model grid creations below
        self.Grid = copy.deepcopy(grid_class)
        self.PSF.set_pixel_size(self.Grid.pixel_width)
        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.ImageNumerics = NumericsSubFrame(pixel_grid=self.Grid, psf=self.PSF, **kwargs_numerics)
        if lens_model_class is None:
            from herculens.LensModel.lens_model import LensModel
            lens_model_class = LensModel(lens_model_list=[])
        self.LensModel = lens_model_class
        if self.LensModel.has_pixels:
            self.Grid.create_model_grid(**self.LensModel.pixel_grid_settings, name='lens')
            self.LensModel.set_pixel_grid(self.Grid.model_pixel_axes('lens'))
        self._psf_error_map = self.PSF.psf_error_map_bool
        if source_model_class is None:
            from herculens.LightModel.light_model import LightModel
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class
        if self.SourceModel.has_pixels:
            self.Grid.create_model_grid(**self.SourceModel.pixel_grid_settings, name='source')
            self.SourceModel.set_pixel_grid(self.Grid.model_pixel_axes('source'), self.Grid.pixel_area)
        if lens_light_model_class is None:
            from herculens.LightModel.light_model import LightModel
            lens_light_model_class = LightModel(light_model_list=[])
        self.LensLightModel = lens_light_model_class
        if self.LensLightModel.has_pixels:
            self.Grid.create_model_grid(**self.LensLightModel.pixel_grid_settings, name='lens_light')
            self.LensLightModel.set_pixel_grid(self.Grid.model_pixel_axes('lens_light'), self.Grid.pixel_area)
        self._kwargs_numerics = kwargs_numerics
        self.source_mapping = Image2SourceMapping(lens_model_class, source_model_class)

    def update_psf(self, psf_class):
        """

        update the instance of the class with a new instance of PSF() with a potentially different point spread function

        :param psf_class:
        :return: no return. Class is updated.
        """
        self.PSF = psf_class
        self.PSF.set_pixel_size(self.Grid.pixel_width)
        self.ImageNumerics = NumericsSubFrame(pixel_grid=self.Grid, psf=self.PSF, **self._kwargs_numerics)
    
    def source_surface_brightness(self, kwargs_source, kwargs_lens=None,
                                  unconvolved=False, de_lensed=False, k=None):
        """

        computes the source surface brightness distribution

        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_extinction: list of keyword arguments of extinction model
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :param k: integer, if set, will only return the model of the specific index
        :return: 2d array of surface brightness pixels
        """
        if len(self.SourceModel.profile_type_list) == 0:
            return np.zeros((self.Grid.num_pixel_axes))
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(ra_grid, dec_grid, kwargs_source, k=k)
        else:
            source_light = self.source_mapping.image_flux_joint(ra_grid, dec_grid, kwargs_lens, kwargs_source, k=k)
        source_light_final = self.ImageNumerics.re_size_convolve(source_light, unconvolved=unconvolved)
        return source_light_final

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """

        computes the lens surface brightness distribution

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :return: 2d array of surface brightness pixels
        """
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(ra_grid, dec_grid, kwargs_lens_light, k=k)
        lens_light_final = self.ImageNumerics.re_size_convolve(lens_light, unconvolved=unconvolved)
        return lens_light_final

    @partial(jit, static_argnums=(0))
    def model(self, kwargs_lens=None, kwargs_source=None,
              kwargs_lens_light=None, unconvolved=False, source_add=True,
              lens_light_add=True):
        """

        make an image from parameter values

        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 2d array of surface brightness pixels of the simulation
        """
        model = np.zeros((self.Grid.num_pixel_axes))
        if source_add is True:
            model += self.source_surface_brightness(kwargs_source, kwargs_lens, unconvolved=unconvolved)
        if lens_light_add is True:
            model += self.lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
        return model

    def simulation(self, add_poisson=True, add_gaussian=True, compute_true_noise_map=True, **model_kwargs):
        """
        same as model() but with noise added

        :param compute_true_noise_map: if True (default), define the noise map (diagonal covariance matrix)
        to be the 'true' one, i.e. based on the noiseless model image.
        """
        if self.Noise is None:
            raise ValueError("Impossible to generate noise realisation because no noise class has been set")
        model = self.model(**model_kwargs)
        noise = self.Noise.realisation(model, add_poisson=add_poisson, add_gaussian=add_gaussian)
        simu = model + noise
        self.Noise.set_data(simu)
        if compute_true_noise_map is True:
            self.Noise.compute_noise_map_from_model(model)
        return simu

    def normalized_residuals(self, data, model, mask=None):
        if mask is None:
            mask = np.ones(self.Grid.num_pixel_axes)
        #noise_var = self.Noise.C_D_model(model)
        noise_var = self.Noise.C_D
        norm_res = (model - data) / np.sqrt(noise_var) * mask
        return norm_res

    def reduced_chi2(self, data, model, mask=None):
        norm_res = self.normalized_residuals(data, model, mask=mask)
        num_data_points = np.count_nonzero(mask)
        return np.sum(norm_res**2) / num_data_points
        