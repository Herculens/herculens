# Defines the model of a multi-plane strong lens
# 
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'krawczyk'

import numpy as np
import jax
import jax.numpy as jnp
import scipy.ndimage

from functools import partial
from herculens.LensImage.Numerics.numerics import Numerics


class MPLensImage(object):
    """Generate a multi-plane lensed images from source light, lens mass/light, and point source models."""

    def __init__(
        self,
        grid_class,
        psf_class,
        noise_class,
        mass_model_class,
        light_model_class,
        source_arc_masks=None,
        source_grid_scale=None,
        conjugate_points=None,
        kwargs_numerics=None
    ):
        """
        :param grid_class: coordinate system, instance of PixelGrid() from herculens.Coordinates.pixel_grid
        :param psf_class: point spread function, instance of PSF() from herculens.Instrument.psf
        :param noise_class: noise properties, instance of Noise() from herculens.Instrument.noise
        :param mass_model_class: multi-plane mass model, instance of MPMassModel() from
            herculens.MassModel.mass_model_multiplane
        :param light_model_class: multi-plane light model, instance of MPLightModel() from
            herculens.LightModel.light_model_multiplane
        :param source_arc_masks: list of 2D boolean array to define the region over which the
            (pixelated) lensed source is modeled, one for each plane
        :param source_grid_scale: a float between 0 and 1 indicating a global scale factor for
            pixelated grids (e.g. 0.5 will set the pixel grid to be 50% the extent defined by
            the arc mask traced back to the source plane)
        :conjugate_points: a list of list of points that can be traced back to each plane
            with the `MPLensImage.trace_conjugate_points` method
        :param kwargs_numerics: keyword arguments for various numerical settings (see
            herculens.Numerics.numerics)
        """
        self.Grid = grid_class
        self.PSF = psf_class
        self.Noise = noise_class
        self.PSF.set_pixel_size(self.Grid.pixel_width)
        assert light_model_class.number_light_planes == mass_model_class.number_mass_planes + 1
        self.MPMassModel = mass_model_class
        self.MPLightModel = light_model_class

        if source_grid_scale is None:
            self._source_grid_scale = [1.0] * self.MPLightModel.number_light_planes
        else:
            self._source_grid_scale = source_grid_scale

        if conjugate_points is None:
            self.conjugate_points = [None] * self.MPMassModel.number_mass_planes
        else:
            self.conjugate_points = conjugate_points

        for i, has_pixels in enumerate(self.MPMassModel.has_pixels):
            if has_pixels:
                pixel_grid = self.Grid.create_model_grid(
                    **self.MPMassModel.mass_models[i].pixel_grid_settings
                )
                self.MPMassModel.mass_models[i].set_pixel_grid(pixel_grid)

        for i, has_pixels in enumerate(self.MPLightModel.has_pixels):
            if has_pixels:
                pixel_grid = self.Grid.create_model_grid(
                    **self.MPLightModel.light_models[i].pixel_grid_settings
                )
                self.MPLightModel.light_models[i].set_pixel_grid(pixel_grid, self.Grid.pixel_area)
        if source_arc_masks is None:
            self.source_arc_masks = np.stack(
                [np.ones(self.Grid.num_pixel_axes)] * self.MPLightModel.number_light_planes
            )
        else:
            self.source_arc_masks = np.stack([
                np.ones(self.Grid.num_pixel_axes) if m is None else m for m in source_arc_masks
            ])

        self._src_adaptive_grid = self.MPLightModel.pixel_is_adaptive

        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.ImageNumerics = Numerics(pixel_grid=self.Grid, psf=self.PSF, **kwargs_numerics)

        ssf = self.ImageNumerics.grid_supersampling_factor

        # get masks in super sampled space
        s_ones = np.ones([ssf, ssf])
        self.source_arc_masks_ss = np.stack([
            np.kron(m, s_ones) for m in self.source_arc_masks
        ])
        # flatten the super sampled masks
        self._source_arc_masks_flat = self.source_arc_masks_ss.reshape(
            self.MPLightModel.number_light_planes,
            -1
        )
        # get the (flattened) outline of the super sampled masks
        # these boundaries are used to define the extent of pixelated grids
        self._source_arc_masks_flat_bool = np.stack([
            (m - scipy.ndimage.binary_erosion(m))
            for m in self.source_arc_masks_ss
        ]).reshape(
            self.MPLightModel.number_light_planes,
            -1
        ).astype(bool)

    def k_extend(self, k, amount):
        if k is None:
            return jnp.arange(amount)
        else:
            return jnp.array(k)

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8))
    def model(
        self,
        eta_flat=None,
        kwargs_mass=None,
        kwargs_light=None,
        supersampled=False,
        k_mass=None,
        k_light=None,
        k_planes=None,
        return_pixel_scale=False
    ):
        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate

        # pixel grid positions on each mass plane (including the lens plane)
        ra_grid_planes, dec_grid_planes = self.MPMassModel.ray_shooting(
            ra_grid_img,
            dec_grid_img,
            eta_flat,
            kwargs_mass,
            k=k_mass
        )
        # (masked) light contribution from each plane
        pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
            ra_grid_planes,
            dec_grid_planes
        )
        light_planes = self.MPLightModel.surface_brightness(
            ra_grid_planes,
            dec_grid_planes,
            kwargs_light,
            pixels_x_coord,
            pixels_y_coord,
            k=k_light,
        ) * self._source_arc_masks_flat
        k_planes = self.k_extend(k_planes, len(light_planes))
        model = light_planes[k_planes].sum(axis=0)
        if not supersampled:
            model = self.ImageNumerics.re_size_convolve(model)
        if return_pixel_scale:
            pixel_scale = [x[1] - x[0] if x is not None else None for x in pixels_x_coord]
            return model, pixel_scale
        else:
            return model

    def simulation(
        self,
        add_poisson=True,
        add_gaussian=True,
        compute_true_noise_map=True,
        noise_seed=18,
        **model_kwargs
    ):
        """
        same as model() but with noise added

        :param compute_true_noise_map: if True (default), define the noise map (diagonal covariance matrix)
        to be the 'true' one, i.e. based on the noiseless model image.
        :param noise_seed: the seed that will be used by the PRNG from JAX to fix the noise realization.
        The default is the arbitrary value 18, so it is the user task to change it for different realizations.
        """
        if self.Noise is None:
            raise ValueError("Impossible to generate noise realization because no noise class has been set")
        model = self.model(**model_kwargs)
        noise = self.Noise.realisation(
            model,
            noise_seed,
            add_poisson=add_poisson,
            add_gaussian=add_gaussian
        )
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

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def trace_conjugate_points(self, eta, kwargs_mass, N=1, k_mass=None):
        i = N - 1
        if self.conjugate_points[i] is not None:
            x, y = self.conjugate_points[i].T
            conj_x, conj_y = self.MPMassModel.ray_shooting(
                x, y,
                eta,
                kwargs_mass,
                k=k_mass,
                N=N,
            )
            return jnp.vstack([conj_x[-1], conj_y[-1]]).T
        else:
            return None

    def mask_extent(self, x_grid_src, y_grid_src, npix_src, source_grid_scale):
        # create grid encompassed by ray-traced coordinates
        x_left, x_right = x_grid_src.min(), x_grid_src.max()
        y_bottom, y_top = y_grid_src.min(), y_grid_src.max()
        # center of the region
        cx = 0.5 * (x_left + x_right)
        cy = 0.5 * (y_bottom + y_top)
        # get the width and height
        width = jnp.abs(x_left - x_right)
        height = jnp.abs(y_bottom - y_top)
        # choose the largest of the two to end up with a square region
        half_size = source_grid_scale * 0.5 * jnp.maximum(height, width)
        # recompute the new boundaries
        x_left = cx - half_size
        x_right = cx + half_size
        y_bottom = cy - half_size
        y_top = cy + half_size
        x_adapt = jnp.linspace(x_left, x_right, npix_src)
        y_adapt = jnp.linspace(y_bottom, y_top, npix_src)
        extent = [x_adapt[0], x_adapt[-1], y_adapt[0], y_adapt[-1]]
        return x_adapt, y_adapt, extent

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def adapt_source_coordinates(
        self,
        ra_grid_planes,
        dec_grid_planes,
        force=False,
        npix_src=100,
        source_grid_scale=1
    ):
        x_adapt = []
        y_adapt = []
        extent_adapt = []
        for i, adapt in enumerate(self._src_adaptive_grid):
            if adapt or force:
                if not force:
                    npix_src, _ = self.MPLightModel.light_models[i].pixel_grid.num_pixel_axes
                    grid_scale = self._source_grid_scale[i]
                else:
                    grid_scale = source_grid_scale
                x_adapt_i, y_adapt_i, extent_i = self.mask_extent(
                    ra_grid_planes[i][self._source_arc_masks_flat_bool[i]],
                    dec_grid_planes[i][self._source_arc_masks_flat_bool[i]],
                    npix_src,
                    grid_scale
                )
                x_adapt.append(x_adapt_i)
                y_adapt.append(y_adapt_i)
                extent_adapt.append(extent_i)
            else:
                x_adapt.append(None)
                y_adapt.append(None)
                extent_adapt.append(None)
        return x_adapt, y_adapt, extent_adapt

    def get_source_coordinates(
        self,
        eta_flat,
        kwargs_mass,
        force=False,
        npix_src=100,
        source_grid_scale=1.0
    ):
        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
        ra_grid_planes, dec_grid_planes = self.MPMassModel.ray_shooting(
            ra_grid_img,
            dec_grid_img,
            eta_flat,
            kwargs_mass
        )
        return self.adapt_source_coordinates(
            ra_grid_planes,
            dec_grid_planes,
            force=force,
            npix_src=npix_src,
            source_grid_scale=source_grid_scale
        )

