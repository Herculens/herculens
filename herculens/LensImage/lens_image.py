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

from herculens.LensImage.Numerics.numerics import Numerics
from herculens.LensImage.lensing_operator import LensingOperator


__all__ = ['LensImage', 'LensImage3D']


class LensImage(object):
    """Generate lensed images from source light, lens mass/light, and point source models."""

    def __init__(self, grid_class, psf_class,
                 noise_class=None, lens_mass_model_class=None,
                 source_model_class=None, lens_light_model_class=None,
                 point_source_model_class=None, 
                 source_arc_mask=None,
                 kwargs_numerics=None,
                 kwargs_lens_equation_solver=None):
        """
        :param grid_class: coordinate system, instance of PixelGrid() from herculens.Coordinates.pixel_grid
        :param psf_class: point spread function, instance of PSF() from herculens.Instrument.psf
        :param noise_class: noise properties, instance of Noise() from herculens.Instrument.noise
        :param lens_mass_model_class: lens mass model, instance of MassModel() from herculens.MassModel.mass_model
        :param source_model_class: extended source light model, instance of LightModel() from herculens.MassModel.mass_model
        :param lens_light_model_class: lens light model, instance of LightModel() from herculens.MassModel.mass_model
        :param point_source_model_class: point source model, instance of PointSourceModel() from herculens.PointSource.point_source
        :param source_arc_mask: 2D boolean array to define the region over which the (pixelated) lensed source is modeled
        :param kwargs_numerics: keyword arguments for various numerical settings (see herculens.Numerics.numerics)
        :param kwargs_lens_equation_solver: TODO
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
            pixel_grid = self.Grid.create_model_grid(
                **self.MassModel.pixel_grid_settings)
            self.MassModel.set_pixel_grid(pixel_grid)

        if source_model_class is None:
            from herculens.LightModel.light_model import LightModel
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class
        if self.SourceModel.has_pixels:
            pixel_grid = self.Grid.create_model_grid(
                **self.SourceModel.pixel_grid_settings)
            self.SourceModel.set_pixel_grid(pixel_grid, self.Grid.pixel_area)

        if lens_light_model_class is None:
            from herculens.LightModel.light_model import LightModel
            lens_light_model_class = LightModel(light_model_list=[])
        self.LensLightModel = lens_light_model_class
        if self.LensLightModel.has_pixels:
            pixel_grid = self.Grid.create_model_grid(
                **self.LensLightModel.pixel_grid_settings)
            self.LensLightModel.set_pixel_grid(
                pixel_grid, self.Grid.pixel_area)

        if point_source_model_class is None:
            from herculens.PointSourceModel.point_source_model import PointSourceModel
            point_source_model_class = PointSourceModel(
                point_source_type_list=[], mass_model=self.MassModel)
        self.PointSourceModel = point_source_model_class
        # self.PointSource.update_lens_model(lens_model_class=lens_model_class)
        # x_center, y_center = self.Data.center
        # self.PointSource.update_search_window(search_window=np.max(self.Data.width), x_center=x_center,
        #                                       y_center=y_center, min_distance=self.Data.pixel_width,
        #                                       only_from_unspecified=True)

        self.source_arc_mask = source_arc_mask
        self._src_adaptive_grid = self.SourceModel.pixel_is_adaptive
        if self._src_adaptive_grid is True and self.source_arc_mask is None:
            raise ValueError("An arc mask for the lensed source must be "
                             "provided with adaptive source grid")
        if self.source_arc_mask is not None:
            self._src_arc_mask_bool = source_arc_mask.astype(bool)

        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.kwargs_numerics = kwargs_numerics
        self.ImageNumerics = Numerics(
            pixel_grid=self.Grid, psf=self.PSF, **self.kwargs_numerics)

        self.kwargs_lens_equation_solver = kwargs_lens_equation_solver

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
            return jnp.zeros(self.Grid.num_pixel_axes)
        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        if self._src_adaptive_grid:
            pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(kwargs_lens, k_lens=k_lens)
        else:
            pixels_x_coord, pixels_y_coord = None, None  # fall back on fixed, user-defined coordinates
        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(x_grid_img, y_grid_img, kwargs_source, k=k,
                                                               pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord)
        else:
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(x_grid_img, y_grid_img, kwargs_lens, k=k_lens)
            source_light = self.SourceModel.surface_brightness(x_grid_src, y_grid_src, kwargs_source, k=k,
                                                               pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord)
        if not supersampled:
            source_light = self.ImageNumerics.re_size_convolve(
                source_light, unconvolved=unconvolved)
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
        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(x_grid_img, y_grid_img, kwargs_lens_light, k=k)
        if not supersampled:
            lens_light = self.ImageNumerics.re_size_convolve(
                lens_light, unconvolved=unconvolved)
        return lens_light

    def point_source_image(self, kwargs_point_source, kwargs_lens,
                           kwargs_solver, k=None):
        """Compute PSF-convolved point sources rendered on the image plane.

        :param kwargs_point_source: list of keyword arguments corresponding to the point sources
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param k: list of bool or list of int to select which point sources to include
        :return: 2d array at the image plane resolution of all multiple images of the point sources
        """
        result = jnp.zeros((self.Grid.num_pixel_axes))
        if self.PointSourceModel is None:
            return result

        theta_x, theta_y, amplitude = self.PointSourceModel.get_multiple_images(
            kwargs_point_source, kwargs_lens, kwargs_solver, k)

        # Extract unique image positions ?
        # def uniquify(a, tol):
        #     return a[~(jnp.triu(jnp.abs(a[:, None] - a) <= tol, 1)).any(0)]

        for i in range(len(theta_x)):
            result += self.ImageNumerics.render_point_sources(
                theta_x[i], theta_y[i], amplitude[i])

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
        if source_add is True:
            source_model = self.source_surface_brightness(kwargs_source, kwargs_lens, 
                                                          unconvolved=unconvolved, supersampled=supersampled,
                                                          k=k_source, k_lens=k_lens) 
            if self.source_arc_mask is not None:
                source_model *= self.source_arc_mask
            model += source_model
        if lens_light_add is True:
            model += self.lens_surface_brightness(kwargs_lens_light, 
                                                  unconvolved=unconvolved, supersampled=supersampled,
                                                  k=k_lens_light)

        if point_source_add:
            model += self.point_source_image(kwargs_point_source, kwargs_lens,
                                             kwargs_solver=self.kwargs_lens_equation_solver,
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
            raise ValueError(
                "Impossible to generate noise realisation because no noise class has been set")
        model = self.model(**model_kwargs)
        noise = self.Noise.realisation(
            model, noise_seed, add_poisson=add_poisson, add_gaussian=add_gaussian)
        simu = model + noise
        self.Noise.set_data(simu)
        if compute_true_noise_map is True:
            self.Noise.compute_noise_map_from_model(model)
        return simu
    
    def C_D_model(self, model):
        return self.Noise.C_D_model(model)

    def normalized_residuals(self, data, model, mask=None):
        """
        compute the map of normalized residuals,
        given the data and the model image
        """
        if mask is None:
            mask = np.ones(self.Grid.num_pixel_axes)
        noise_var = self.C_D_model(model)
        noise = np.sqrt(noise_var)
        norm_res_model = (data - model) / noise * mask
        norm_res_tot = norm_res_model
        if mask is not None:
            # outside the mask just add pure data
            norm_res_tot += (data / noise) * (1. - mask)
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
    
    def adapt_source_coordinates(self, kwargs_lens, k_lens=None, return_plt_extent=False):
        """
        Compute new source coordinates based on ray-traced arc-mask
        return_plt_extent=True returns the extent of the grid for imshow
        """
        # TODO: compute a mask on the adapted source plane
        npix_src, npix_src_y = self.SourceModel.pixel_grid.num_pixel_axes
        if npix_src_y != npix_src:
            raise ValueError("Adaptive source plane grid only works with square grids")
        if self.Grid.x_is_inverted or self.Grid.y_is_inverted:
            # TODO: implement this
            raise NotImplementedError("invert x and y not yet supported for adaptive source grid")
        # get data coordinates
        x_grid_img, y_grid_img = self.Grid.pixel_coordinates
        # ray-shoot to source plane only coordinates within the source arc mask
        x_grid_src, y_grid_src = self.MassModel.ray_shooting(x_grid_img[self._src_arc_mask_bool], 
                                                             y_grid_img[self._src_arc_mask_bool], 
                                                             kwargs_lens, k=k_lens)
        # create grid encompassed by ray-shot coordinates
        x_left, x_right = x_grid_src.min(), x_grid_src.max()
        y_bottom, y_top = y_grid_src.min(), y_grid_src.max()
        # center of the region
        cx = 0.5 * (x_left + x_right)
        cy = 0.5 * (y_bottom + y_top)
        # get the width and height
        width  = jnp.abs(x_left - x_right)
        height = jnp.abs(y_bottom - y_top)
        # choose the largest of the two to end up with a square region
        half_size = jnp.maximum(height, width) / 2.
        # recompute the new boundaries
        x_left = cx - half_size
        x_right = cx + half_size
        y_bottom = cy - half_size
        y_top = cy + half_size
        # print( 0.5*(x_left + x_right), cx )
        # print( jnp.abs(x_left - x_right), size )
        # print( 0.5*(y_bottom + y_top), cy )
        # print( jnp.abs(y_bottom - y_top), size )
        x_adapt = jnp.linspace(x_left, x_right, npix_src)
        y_adapt = jnp.linspace(y_bottom, y_top, npix_src)
        if return_plt_extent is True:
            pix_scl_x = jnp.abs(x_adapt[0]-x_adapt[1])
            pix_scl_y = jnp.abs(y_adapt[0]-y_adapt[1])
            half_pix_scl = jnp.sqrt(pix_scl_x*pix_scl_y) / 2.
            extent_adapt = [
                x_adapt[0]-half_pix_scl, x_adapt[-1]+half_pix_scl, 
                y_adapt[0]-half_pix_scl, y_adapt[-1]+half_pix_scl
            ]
        else:
            extent_adapt = [
                x_adapt[0], x_adapt[-1], 
                y_adapt[0], y_adapt[-1]
            ]
        return x_adapt, y_adapt, extent_adapt
    
    def get_source_coordinates(self, kwargs_lens, k_lens=None, return_plt_extent=False):
        if not self._src_adaptive_grid:
            x_grid, y_grid = self.SourceModel.pixel_grid.pixel_coordinates
            if return_plt_extent is True:
                extent = self.SourceModel.pixel_grid.plt_extent
            else:
                extent = self.SourceModel.pixel_grid.extent
        else:
            x_coord, y_coord, extent = self.adapt_source_coordinates(kwargs_lens, k_lens=k_lens,
                                                                     return_plt_extent=return_plt_extent)
            x_grid, y_grid = np.meshgrid(x_coord, y_coord)
        return x_grid, y_grid, extent
        
    def get_lensing_operator(self, kwargs_lens=None, update=False, arc_mask=None):
        if self.SourceModel.pixel_grid is None:
            raise ValueError("The lensing operator is only defined for source "
                            "models associated to a grid of pixels")
        if not hasattr(self, '_lensing_op'):
            self._lensing_op = LensingOperator(self.MassModel,
                                            self.Grid, # TODO: should be the model grid (from Numerics) at some point
                                            self.SourceModel.pixel_grid,
                                            arc_mask=arc_mask)
        if update is True or self._lensing_op.get_lens_mapping() is None:
            self._lensing_op.compute_mapping(kwargs_lens)
        return self._lensing_op
    


class LensImage3D(object):
    """Generate lensed images from source light, lens mass/light, and point source models."""

    def __init__(self, lens_image_list):
        """
        :param lens_image_list: list of LensImage instances
        """
        self.num_bands = len(lens_image_list)
        for i in range(1, self.num_bands):
            if lens_image_list[i].Grid.num_pixel_axes != lens_image_list[i-1].Grid.num_pixel_axes:
                raise ValueError("In each band, all models should have the same shape.")
        self.lens_images = lens_image_list
        self.nx, self.ny = lens_image_list[0].Grid.num_pixel_axes
        self.nw = self.num_bands

    def source_surface_brightness(self, kwargs_source_list, k_list=None, **kwargs_single_band):
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
        if k_list is None:
            k_list = [None]*self.num_bands
        source_light_list = []
        for i in range(self.num_bands):
            source_light = self.lens_images[i].source_surface_brightness(
                kwargs_source_list[i], k=k_list[i], **kwargs_single_band,
            )
            source_light_list.append(source_light)
        return source_light_list

    def lens_surface_brightness(self, kwargs_lens_light_list, k_list=None, **kwargs_single_band):
        """

        computes the lens surface brightness distribution

        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :param k: list of bool or list of int to select which model profiles to include
        :return: 2d array of surface brightness pixels
        """
        if k_list is None:
            k_list = [None]*self.num_bands
        lens_light_list = []
        for i in range(self.num_bands):
            lens_light = self.lens_images[i].lens_surface_brightness(
                kwargs_lens_light_list[i], k=k_list[i], **kwargs_single_band,
            )
            lens_light_list.append(lens_light)
        return lens_light_list

    def point_source_image(self, kwargs_point_source_list, k_list=None, **kwargs_single_band):
        """Compute PSF-convolved point sources rendered on the image plane.

        :param kwargs_point_source: list of keyword arguments corresponding to the point sources
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param k: list of bool or list of int to select which point sources to include
        :return: 2d array at the image plane resolution of all multiple images of the point sources
        """
        if k_list is None:
            k_list = [None]*self.num_bands
        point_source_list = []
        for i in range(self.num_bands):
            point_source = self.lens_images[i].point_source_image(
                kwargs_point_source_list[i], k=k_list[i], **kwargs_single_band,
            )
            point_source_list.append(point_source)
        return point_source_list

    @partial(jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    def model(self, kwargs_lens=None, kwargs_source_list=None, kwargs_lens_light_list=None,
            kwargs_point_source_list=None, unconvolved=False, supersampled=False,
            source_add=True, lens_light_add=True, point_source_add=True,
            k_lens=None, k_source_list=None, k_lens_light_list=None, k_point_source_list=None):
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
        kwargs_source_list = self._none_kwargs(kwargs_source_list)
        kwargs_lens_light_list = self._none_kwargs(kwargs_lens_light_list)
        kwargs_point_source_list = self._none_kwargs(kwargs_point_source_list)
        k_source_list = self._none_kwargs(k_source_list)
        k_lens_light_list = self._none_kwargs(k_lens_light_list)
        k_point_source_list = self._none_kwargs(k_point_source_list)
        model_multi_band = []
        for i in range(self.num_bands):
            model_sgl_band = self.lens_images[i].model(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source_list[i],
                kwargs_lens_light=kwargs_lens_light_list[i],
                kwargs_point_source=kwargs_point_source_list[i],
                unconvolved=unconvolved, supersampled=supersampled,
                source_add=source_add, lens_light_add=lens_light_add, 
                point_source_add=point_source_add,
                k_lens=k_lens, k_lens_light=k_lens_light_list[i], k_source=k_source_list[i], 
                k_point_source=k_point_source_list[i],
            )
            model_multi_band.append(model_sgl_band)
        return jnp.array(model_multi_band)

    def simulation(self, add_poisson=True, add_gaussian=True,
                compute_true_noise_map=True, noise_seed=18,
                **model_kwargs):
        raise NotImplementedError()
    
    def C_D_model(self, model):
        c_d_multi_band = []
        for i in range(self.num_bands):
            c_d_slg_band = self.lens_images[i].C_D_model(model[i])
            c_d_multi_band.append(c_d_slg_band)
        return jnp.array(c_d_multi_band)

    def normalized_residuals(self, data, model, mask=None):
        """
        compute the map of normalized residuals,
        given the data and the model image
        """
        if mask is None:
            mask = np.ones_like(data)
        noise_var = self.C_D_model(model)
        noise = np.sqrt(noise_var)
        norm_res_model = (data - model) / noise * mask
        norm_res_tot = norm_res_model
        if mask is not None:
            # outside the mask just add pure data
            norm_res_tot += (data / noise) * (1. - mask)
        return norm_res_model, norm_res_tot

    def reduced_chi2(self, data, model, mask_list=None):
        """
        compute the reduced chi2 of the data given the model
        """
        if mask is None:
            mask = np.ones_like(data)
        norm_res, _ = self.normalized_residuals(data, model, mask=mask)
        num_data_points = np.sum(mask)
        return np.sum(norm_res**2) / num_data_points
    
    def _none_kwargs(self, kwargs):
        return kwargs if kwargs is not None else [None]*self.num_bands
    