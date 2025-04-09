# Class to plot a lens model
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import copy
import warnings
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, ArrowStyle

from herculens.LensImage.lens_image import LensImage, LensImage3D
from herculens.LensImage.lens_image_multiplane import MPLensImage
from herculens.Util.plot_util import nice_colorbar, nice_colorbar_residuals, contour_with_legend
from herculens.Util import model_util


# Some general default for plotting
plt.rc('image', interpolation='none', origin='lower')  # for imshow


__all__ = ['Plotter']


class Plotter(object):
    """
    Helper class to plot the results of a LensImage model.

    Parameters
    ----------
    data_name : str, optional
        The name of the data, by default None.
    base_fontsize : int, optional
        The base fontsize for the plot, by default 14.
    flux_log_scale : bool, optional
        Whether to use a logarithmic scale for the flux, by default True.
    flux_vmin : float, optional
        The minimum value for the flux, by default None.
    flux_vmax : float, optional
        The maximum value for the flux, by default None.
    res_vmax : int, optional
        The maximum value for the residual, by default 6.
    cmap_flux : str, optional
        The colormap for the flux, by default None.
    """

    # Define some custom colormaps
    cmap_flux = copy.copy(plt.get_cmap('magma'))
    cmap_flux.set_under('black')
    cmap_flux.set_over('white')
    cmap_flux.set_bad('black')
    cmap_flux_alt = copy.copy(cmap_flux)
    cmap_flux_alt.set_bad('#222222')  # to emphasize non-positive pixels in log scale
    cmap_res = plt.get_cmap('RdBu_r')
    cmap_corr = plt.get_cmap('RdYlGn')
    cmap_default = plt.get_cmap('viridis')
    cmap_deriv1 = plt.get_cmap('cividis')
    cmap_deriv2 = plt.get_cmap('inferno')
    cmap_bw = copy.copy(plt.get_cmap('gray'))
    cmap_bw.set_under('black')
    cmap_bw.set_over('white')
    cmap_bw.set_bad('black')

    def __init__(self, data_name=None, base_fontsize=14, flux_log_scale=True, 
                 flux_vmin=None, flux_vmax=None, res_vmax=6, cmap_flux=None,
                 ref_lens_image=None, ref_kwargs_result=None):
        self.data_name = data_name
        self.base_fontsize = base_fontsize
        self.flux_log_scale = flux_log_scale
        if self.flux_log_scale is True:
            self.norm_flux = LogNorm(flux_vmin, flux_vmax)
        else:
            self.norm_flux = None
        self.norm_res = Normalize(-res_vmax, res_vmax)
        self.norm_corr = TwoSlopeNorm(0)
        if cmap_flux is not None:
            self.cmap_flux = cmap_flux
            self.cmap_flux_alt = cmap_flux
        if ref_lens_image is not None and ref_kwargs_result is None:
            raise ValueError("If a reference lens image is provided, "
                             "the reference kwargs_result must also be provided.")
        self.ref_lens_image = ref_lens_image
        self.ref_kwargs_result = ref_kwargs_result

    def set_data(self, data):
        self._data = data

    def set_ref_source(self, ref_source, plt_extent=None):
        if self.ref_lens_image is not None:
            raise ValueError("Reference source already set from a LensImage instance.")
        self._ref_source = ref_source
        self._ref_source_extent = plt_extent

    def set_ref_lens_light(self, ref_lens_light):
        self._ref_lens_light = ref_lens_light

    def set_ref_pixelated_potential(self, ref_potential):
        self._ref_pixel_pot = ref_potential

    def plot_flux(self, image, title=None):
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(image, extent=None, cmap=self.cmap_flux, norm=self.norm_flux)
        if title is not None:
            ax.set_title(title, fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        return fig

    def model_summary(
            self, lens_image, kwargs_result,
            show_image=True, show_source=True, 
            show_lens_light=False, show_lens_potential=False, show_lens_others=False,
            only_pixelated_potential=False, shift_pixelated_potential='none',
            likelihood_mask=None, potential_mask=None, 
            show_lens_lines=False, show_shear_field=False, show_lens_position=False,
            kwargs_grid_source=None,
            lock_colorbars=False, masked_residuals=True,
            vmin_pot=None, vmax_pot=None,
            k_source=None, k_lens=None, k_lens_light=None, k_point_source=None,  # for single-plane models
            k_light=None, k_mass=None, k_planes=None,  # for multi-plane models
            kwargs_noise=None, 
            figsize=None, show_plot=True,

            # TODO: this is a dirty quick fix to correctly read multi-band heterogenous LensImage models
            # This will not be necessary once the LensImage3D class is properly supported.
            adapted_source_pixels_coords=None,
        ):
        """
        Generate a summary plot of the lens model.

        Parameters
        ----------
        lens_image : LensImage
            A herculens.LensImage instance.
        kwargs_result : dict
            Nested dictionary containing all model parameters.
        show_image : bool, optional
            Whether to show the lens image, by default True.
        show_source : bool, optional
            Whether to show the source model, by default True.
        show_lens_light : bool, optional
            Whether to show the lens light model, by default False.
        show_lens_potential : bool, optional
            Whether to show the lens potential model, by default False.
        show_lens_others : bool, optional
            Whether to show other lens models, by default False.
        only_pixelated_potential : bool, optional
            Whether to show only the pixelated potential, by default False.
        shift_pixelated_potential : str, optional
            The type of shift to apply to the pixelated potential, by default 'none'.
        likelihood_mask : ndarray, optional
            The likelihood mask, by default None.
        potential_mask : ndarray, optional
            The potential mask, by default None.
        show_lens_lines : bool, optional
            Whether to show lens lines, by default False.
        show_shear_field : bool, optional
            Whether to show the shear field, by default False.
        show_lens_position : bool, optional
            Whether to show the lens position, by default False.
        kwargs_grid_source : dict, optional
            The grid source parameters, by default None.
        lock_colorbars : bool, optional
            Whether to lock the colorbars, by default False.
        masked_residuals : bool, optional
            Whether to show masked residuals, by default True.
        vmin_pot : float, optional
            The minimum potential value, by default None.
        vmax_pot : float, optional
            The maximum potential value, by default None.
        k_lens : float, optional
            The lens model normalization factor, by default None.
        kwargs_noise : dict, optional
            Parameters given to Noise.C_D_model(mode, **kwargs_noise), by default None.
        show_plot : bool, optional
            Whether to call plt.show(), by default True.

        Returns
        -------
        type
            The summary plot.
        """
        if isinstance(lens_image, MPLensImage) and any([
            show_source, show_lens_light, show_lens_potential, show_lens_others,
            show_shear_field, show_lens_position, show_lens_lines,
        ]):
            raise NotImplementedError("The full plotting of multi-plane lens models, "
                                      "other than just the data model (observed image plane), "
                                      "is not supported yet.")
            
        n_cols = 3
        n_rows = sum([show_image, show_source, show_lens_light, 
                      show_lens_potential, show_lens_others])
        
        # extent = lens_image.Grid.extent
        extent = lens_image.Grid.plt_extent

        ##### PREPARE IMAGES #####

        if kwargs_noise is None:
            kwargs_noise = {}
            
        if show_image:
            # create the resulting model image
            if isinstance(lens_image, (LensImage, LensImage3D)):
                model = lens_image.model(
                    **kwargs_result, 
                    k_lens=k_lens,
                    k_lens_light=k_lens_light,
                    k_source=k_source,
                    k_point_source=k_point_source,
                    adapted_source_pixels_coords=adapted_source_pixels_coords,
                )
            elif isinstance(lens_image, MPLensImage):
                model = lens_image.model(
                    **kwargs_result, 
                    k_mass=k_mass,
                    k_light=k_light,
                    k_planes=k_planes,
                )
            if likelihood_mask is None:
                mask_bool = False
                likelihood_mask = np.ones_like(model)
            else:
                mask_bool = True
            # create a mask with NaNs such that unmasked areasa appear transparent 
            likelihood_mask_nans = np.nan*np.copy(likelihood_mask)
            likelihood_mask_nans[likelihood_mask == 0] = 0

            if hasattr(self, '_data'):
                data = self._data
            else:
                data = np.zeros_like(model)

        if show_source:
            kwargs_source = copy.deepcopy(kwargs_result['kwargs_source'])
            if lens_image.SourceModel.has_pixels or kwargs_grid_source is not None:
                if kwargs_grid_source is not None:
                    grid_src = lens_image.Grid.create_model_grid(**kwargs_grid_source)
                    x_grid_src, y_grid_src = grid_src.pixel_coordinates
                    src_extent = grid_src.plt_extent
                else:
                    x_grid_src, y_grid_src, src_extent = lens_image.get_source_coordinates(
                        kwargs_result['kwargs_lens'], k_lens=k_lens, return_plt_extent=True
                    )
                source_model = lens_image.eval_source_surface_brightness(
                    x_grid_src, y_grid_src, 
                    kwargs_source, kwargs_lens=kwargs_result['kwargs_lens'],
                    k=k_source, k_lens=k_lens, de_lensed=True,
                    adapted_pixels_coords=adapted_source_pixels_coords,
                ) * lens_image.Grid.pixel_area
            else:
                source_model = lens_image.source_surface_brightness(
                    kwargs_source, kwargs_lens=kwargs_result['kwargs_lens'], 
                    de_lensed=True, unconvolved=True, 
                    k=k_source, k_lens=k_lens,
                )
                x_grid_src, y_grid_src = lens_image.ImageNumerics.coordinates_evaluate
                src_extent = extent

            if self.ref_lens_image is not None:
                ref_source = self.ref_lens_image.eval_source_surface_brightness(
                    x_grid_src, y_grid_src, 
                    self.ref_kwargs_result['kwargs_source'], 
                    de_lensed=True, 
                    adapted_pixels_coords=adapted_source_pixels_coords,
                ) * lens_image.Grid.pixel_area
                ref_src_extent = src_extent
                show_source_diff = True
            elif hasattr(self, '_ref_source'):
                ref_source = self._ref_source
                if source_model.shape != ref_source.shape:
                    warnings.warn("Reference source does not have the same shape as model source.")
                    show_source_diff = False
                else:
                    show_source_diff = True
                ref_src_extent = self._ref_source_extent
            else:
                ref_source = None
                ref_src_extent = None
                show_source_diff = False

            if len(lens_image.PointSourceModel.type_list) > 0:
                #TODO: support several point source models
                ps0_params = kwargs_result['kwargs_point_source'][0]
                all_ps_src_x, all_ps_src_y = lens_image.PointSourceModel.get_source_plane_points(
                    kwargs_result['kwargs_point_source'],
                    kwargs_lens=kwargs_result['kwargs_lens'],
                    with_amplitude=False,
                )
                ps_src_pos = (all_ps_src_x[0], all_ps_src_y[0])
            else:
                ps_src_pos = None

        if show_lens_light:
            kwargs_lens_light = copy.deepcopy(kwargs_result['kwargs_lens_light'])
            if lens_image.LensLightModel.has_pixels:
                ll_idx = lens_image.LensLightModel.pixelated_index
                lens_light_model = kwargs_lens_light[ll_idx]['pixels']
            else:
                lens_light_model = lens_image.lens_surface_brightness(kwargs_lens_light, unconvolved=True)
            
            if hasattr(self, '_ref_lens_light'):
                ref_lens_light = self._ref_lens_light
                if lens_light_model.shape != ref_lens_light.shape:
                    warnings.warn("Reference lens light does not have the same shape as model lens light.")
                    show_lens_light_diff = False
                else:
                    show_lens_light_diff = True
            else:
                ref_lens_light = None
                show_lens_light_diff = False

        if show_lens_potential or show_lens_others:
            kwargs_lens = copy.deepcopy(kwargs_result['kwargs_lens'])
            pot_idx = lens_image.MassModel.pixelated_index if only_pixelated_potential else None
            if pot_idx is not None and only_pixelated_potential:
                x_grid_lens, y_grid_lens = lens_image.MassModel.pixel_grid.pixel_coordinates
                potential_model = kwargs_lens[pot_idx]['pixels']
            else:
                x_grid_lens, y_grid_lens = lens_image.Grid.pixel_coordinates
                if show_lens_potential:
                    potential_model = lens_image.MassModel.potential(x_grid_lens, y_grid_lens, 
                                                                     kwargs_lens, k=pot_idx)
            alpha_x, alpha_y = lens_image.MassModel.alpha(x_grid_lens, y_grid_lens, 
                                                          kwargs_lens, k=pot_idx)
            kappa = lens_image.MassModel.kappa(x_grid_lens, y_grid_lens, 
                                               kwargs_lens, k=pot_idx)
            #kappa = ndimage.gaussian_filter(kappa, 1)
            magnification = lens_image.MassModel.magnification(x_grid_lens, y_grid_lens, kwargs_lens)
            
            if potential_mask is None:
                potential_mask = np.ones_like(x_grid_lens)

            # here we know that there are no perturbations in the reference potential
            if hasattr(self, '_ref_pixel_pot') and show_lens_potential:
                ref_potential = self._ref_pixel_pot
                if ref_potential.shape != potential_model.shape:
                    warnings.warn("Reference potential does not have the same shape as model potential.")
                    show_pot_diff = False
                else:
                    show_pot_diff = True
            
                if shift_pixelated_potential == 'min':
                    min_in_mask = potential_model[potential_mask == 1].min()
                    potential_model = potential_model - min_in_mask
                    ref_min_in_mask = ref_potential[potential_mask == 1].min()
                    ref_potential = ref_potential - ref_min_in_mask
                    print("delta_psi shift by min:", min_in_mask)
                if shift_pixelated_potential == 'max':
                    max_in_mask = potential_model[potential_mask == 1].max()
                    potential_model = potential_model - max_in_mask
                    ref_max_in_mask = ref_potential[potential_mask == 1].max()
                    ref_potential = ref_potential - ref_max_in_mask
                    print("delta_psi shift by max:", max_in_mask)
                elif shift_pixelated_potential == 'mean':
                    mean_in_mask = potential_model[potential_mask == 1].mean()
                    potential_model = potential_model - mean_in_mask
                    ref_mean_in_mask = ref_potential[potential_mask == 1].mean()
                    ref_potential = ref_potential - ref_mean_in_mask
                    print("delta_psi shift by mean values:", mean_in_mask, ref_mean_in_mask)

            else:
                ref_potential = None
                show_pot_diff = False

        if show_lens_lines:
            clines, caustics, centers = model_util.critical_lines_caustics(
                lens_image, kwargs_result['kwargs_lens'], return_lens_centers=True)


        ##### BUILD UP THE PLOTS #####
        if figsize is None:
            figsize = (15, n_rows*5)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if len(axes.shape) == 1:
            axes = axes[None, :] # add first axis so axes is always 2d array
        i_row = 0

        if show_image:
            norm_flux = self._get_norm_for_model(model, lock_colorbars)

            ##### IMAGING DATA AND MODEL IMAGE #####
            ax = axes[i_row, 0]
            im = ax.imshow(data, extent=extent, cmap=self.cmap_flux, norm=norm_flux)
            im.set_rasterized(True)
            if mask_bool is True:
                ax.contour(likelihood_mask, extent=extent, levels=[0], 
                           colors='white', alpha=0.3, linewidths=0.5)
            if show_lens_position and 'kwargs_lens_light' in kwargs_result:
                plotted_centers = []
                for kw in kwargs_result['kwargs_lens_light']:
                    if 'center_x' in kw:
                        center = (kw['center_x'], kw['center_y'])
                        if center not in plotted_centers:
                            ax.plot(*center, linestyle='none', color='black', 
                                    markeredgecolor='black', markersize=15, marker='o', 
                                    fillstyle='none', markeredgewidth=1)
                        plotted_centers.append(center)
            if show_lens_lines:
                for curve in clines:
                    ax.plot(curve[0], curve[1], linewidth=0.8, color='white')
                ax.scatter(*centers, s=20, c='gray', marker='+', linewidths=0.5)
            if show_shear_field:
                shear_field = model_util.shear_deflection_field(lens_image, kwargs_result['kwargs_lens'], num_pixels=8)
                if shear_field is not None:
                    x_field, y_field, gx_field, gy_field = shear_field
                    ax.quiver(
                        x_field, y_field, # + overall_shift, 
                        gx_field, gy_field, 
                        scale=0.2, 
                        width=0.05,
                        scale_units='xy', units='xy',
                        pivot='middle',
                        headaxislength=0, headlength=0,
                        color='white', alpha=0.3,
                    )
                    ax.set_xlim(extent[0], extent[1])
                    ax.set_ylim(extent[2], extent[3])
                else:
                    print("Warning: no external shear to plot have been found.")
            data_title = self.data_name if self.data_name is not None else "data"
            ax.set_title(data_title, fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            
            ax = axes[i_row, 1]
            im = ax.imshow(model, extent=extent, cmap=self.cmap_flux, norm=norm_flux)
            im.set_rasterized(True)
            ax.set_title("model", fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})

            ax = axes[i_row, 2]
            model_residuals, residuals = lens_image.normalized_residuals(
                data, model, kwargs_noise=kwargs_noise, mask=likelihood_mask,
            )
            if masked_residuals is True:
                residuals_plot = model_residuals
            else:
                residuals_plot = residuals
            red_chi2 = lens_image.reduced_chi2(
                data, model, kwargs_noise=kwargs_noise, mask=likelihood_mask,
            )
            im = ax.imshow(residuals_plot, cmap=self.cmap_res, extent=extent, norm=self.norm_res)
            im.set_rasterized(True)
            if mask_bool is True and masked_residuals is False:
                ax.contour(likelihood_mask, extent=extent, levels=[0], 
                           colors='black', alpha=0.5, linewidths=0.5)
            ax.set_title(r"(f${}_{\rm data}$ - f${}_{\rm model})/\sigma$", fontsize=self.base_fontsize)
            nice_colorbar_residuals(im, residuals_plot, position='top', pad=0.4, size=0.2, 
                                    vmin=self.norm_res.vmin, vmax=self.norm_res.vmax,
                                    colorbar_kwargs={'orientation': 'horizontal'})
            text = r"$\chi^2_\nu={:.2f}$".format(red_chi2)
            ax.text(0.05, 0.05, text, color='black', fontsize=self.base_fontsize-4, 
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8})
            i_row += 1

        if show_source:
            norm_flux = self._get_norm_for_model(source_model, lock_colorbars)

            ##### UNLENSED AND UNCONVOLVED SOURCE MODEL #####
            ax = axes[i_row, 0]
            if ref_source is not None:
                im = ax.imshow(ref_source, extent=ref_src_extent, cmap=self.cmap_flux_alt, norm=norm_flux) #, vmax=vmax)
                im.set_rasterized(True)
                ax.set_title("ref. source", fontsize=self.base_fontsize)
                nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                              colorbar_kwargs={'orientation': 'horizontal'})
            else:
                ax.axis('off')
            ax = axes[i_row, 1]
            im = ax.imshow(source_model, extent=src_extent, cmap=self.cmap_flux_alt, norm=norm_flux) #, vmax=vmax)
            #im = ax.imshow(source_model, extent=extent, cmap=self.cmap_flux_alt, norm=LogNorm(1e-5))
            im.set_rasterized(True)
            ax.set_title("source model", fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            if show_lens_lines:
                for curve in caustics:
                    ax.plot(curve[0], curve[1], linewidth=0.8, color='white')
                    # force x, y limits to stay the same as 
                    ax.set_xlim(src_extent[0], src_extent[1])
                    ax.set_ylim(src_extent[2], src_extent[3])
            if ps_src_pos is not None:
                ax.scatter(*ps_src_pos, s=100, c='tab:green', marker='*', linewidths=0.5, 
                           label="point source")
                ax.legend()
            ax = axes[i_row, 2]
            if ref_source is not None and show_source_diff is True:
                diff = source_model - ref_source
                vmax_diff = ref_source.max() / 10.
                im = ax.imshow(diff, extent=src_extent, 
                               cmap=self.cmap_res, norm=Normalize(-vmax_diff, vmax_diff))
                im.set_rasterized(True)
                ax.set_title(r"s${}_{\rm model}$ - s${}_{\rm ref}$", fontsize=self.base_fontsize)
                nice_colorbar_residuals(im, diff, position='top', pad=0.4, size=0.2, 
                                        vmin=-vmax_diff, vmax=vmax_diff,
                                        colorbar_kwargs={'orientation': 'horizontal'})
            else:
                ax.axis('off')
            i_row += 1

        if show_lens_light:
            norm_flux = self._get_norm_for_model(lens_light_model, lock_colorbars)

            ##### UNLENSED AND UNCONVOLVED SOURCE MODEL #####
            ax = axes[i_row, 0]
            if ref_lens_light is not None:
                im = ax.imshow(ref_lens_light, extent=extent, cmap=self.cmap_flux_alt, norm=norm_flux) #, vmax=vmax)
                im.set_rasterized(True)
                ax.set_title("ref. lens light", fontsize=self.base_fontsize)
                nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                              colorbar_kwargs={'orientation': 'horizontal'})
            else:
                ax.axis('off')
            ax = axes[i_row, 1]
            im = ax.imshow(lens_light_model, extent=extent, cmap=self.cmap_flux_alt, norm=norm_flux) #, vmax=vmax)
            #im = ax.imshow(lens_light_model, extent=extent, cmap=self.cmap_flux_alt, norm=LogNorm(1e-5))
            im.set_rasterized(True)
            ax.set_title("lens light model", fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            ax = axes[i_row, 2]
            if ref_lens_light is not None:
                diff = lens_light_model - ref_lens_light
                vmax_diff = ref_lens_light.max() / 10.
                im = ax.imshow(diff, extent=extent, 
                               cmap=self.cmap_res, norm=Normalize(-vmax_diff, vmax_diff))
                im.set_rasterized(True)
                ax.set_title(r"l${}_{\rm model}$ - l${}_{\rm ref}$", fontsize=self.base_fontsize)
                nice_colorbar_residuals(im, diff, position='top', pad=0.4, size=0.2, 
                                        vmin=-vmax_diff, vmax=vmax_diff,
                                        colorbar_kwargs={'orientation': 'horizontal'})
            else:
                ax.axis('off')
            i_row += 1

        if show_lens_potential:

            ##### PIXELATED POTENTIAL PERTURBATIONS #####
            ax = axes[i_row, 0]
            if ref_potential is not None:
                im = ax.imshow(ref_potential * potential_mask, extent=extent,
                               vmin=vmin_pot, vmax=vmax_pot,
                               cmap=self.cmap_default)
                im.set_rasterized(True)
                ax.set_title(r"$\psi_{\rm pix, ref}$", fontsize=self.base_fontsize)
                nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                              colorbar_kwargs={'orientation': 'horizontal'})
                ax.imshow(likelihood_mask_nans, extent=extent, cmap='gray_r', vmin=0, vmax=1)
            else:
                ax.axis('off')

            ax = axes[i_row, 1]
            im = ax.imshow(potential_model * potential_mask, extent=extent,
                           vmin=vmin_pot, vmax=vmax_pot,
                           cmap=self.cmap_default)
            ax.set_title(r"$\psi_{\rm pix}$", fontsize=self.base_fontsize)
            im.set_rasterized(True)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            ax.imshow(likelihood_mask_nans, extent=extent, cmap='gray_r', vmin=0, vmax=1)

            ax = axes[i_row, 2]
            if ref_potential is not None and show_pot_diff is True:
                pot_abs_res = (ref_potential - potential_model) * potential_mask
                vmax = np.max(np.abs(ref_potential)) / 2.
                im = ax.imshow(pot_abs_res, extent=extent,
                               vmin=-vmax, vmax=vmax,
                               cmap=self.cmap_res)
                ax.set_title(r"$\psi_{\rm pix}$ - $\psi_{\rm pix, ref}$", fontsize=self.base_fontsize)
                nice_colorbar_residuals(im, pot_abs_res, position='top', pad=0.4, size=0.2, 
                                        vmin=-vmax, vmax=vmax,
                                        colorbar_kwargs={'orientation': 'horizontal'})
                im.set_rasterized(True)
                ax.imshow(likelihood_mask_nans, extent=extent, cmap='gray_r', vmin=0, vmax=1)
            else:
                ax.axis('off')
            i_row += 1

        if show_lens_others:

            ##### DEFLECTION ANGLES, SURFACE MASS DENSITY AND MAGNIFICATION #####
            ax = axes[i_row, 0]
            im = ax.imshow(alpha_x * potential_mask, cmap=self.cmap_deriv1, alpha=1, extent=extent)
            im.set_rasterized(True)
            title = r"$\alpha_{x,\rm pix}$" if only_pixelated_potential else r"deflection field ($x$)"
            ax.set_title(title, fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            ax = axes[i_row, 1]
            im = ax.imshow(kappa * potential_mask, cmap=self.cmap_deriv2, norm=LogNorm(),
                           alpha=1, extent=extent)
            im.set_rasterized(True)
            title = r"$\kappa_{\rm pix}$" if only_pixelated_potential else "convergence"
            ax.set_title(title, fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            ax = axes[i_row, 2]
            im = ax.imshow(magnification * potential_mask, cmap=self.cmap_default, norm=Normalize(-10, 10),
                           alpha=1, extent=extent)
            im.set_rasterized(True)
            title = r"$\mu_{\rm pix}$" if only_pixelated_potential else "magnification"
            ax.set_title(title, fontsize=self.base_fontsize)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            i_row += 1

        if show_plot:
            plt.show()
        return fig
    

    def model_summary_multiplane(
            self,
            lens_image,
            kwargs_result,
            kwargs_noise=None,
            likelihood_mask=None,
            masked_residuals=True,
            show_shear_field=False,
            norm_kappa=LogNorm(1e-1, 1e1),
            extent_zoom=[-0.5, 0.5, -0.5, 0.5],
            kwargs_grid_source=None,
            linestyles_planes=[':', '-', '--', '-.', ':', '-'],
            colors_planes=['tab:purple', 'tab:cyan', 'tab:orange', 'tab:pink', 'tab:blue', 'tab:red'],
        ):
        """
        Simple function with limited user-control to plot the details
        of a multi-plane lens model based on a MPLensImage instance.

        NOTE: This function will likely improve / be heavily revamped in the future.

        Here `kwargs_grid_source` can also be a list for each source plane
        """
        if not isinstance(lens_image, MPLensImage):
            raise ValueError("This function is only for multi-plane lens models.")
        if kwargs_noise is None:
            kwargs_noise = {}
        
        # get the total number of planes as we will iterate over it
        num_planes = lens_image.MPLightModel.number_light_planes

        # optional grid parameters of sources for each plane
        if not isinstance(kwargs_grid_source, (list, tuple)):
            kwargs_grid_source = [copy.deepcopy(kwargs_grid_source) for _ in range(num_planes - 1)]

        # create a figure with multiple subplots:
        # - the first row contains the data, the model and the normalized residuals, and the model with critical lines overlayed
        # - the second row contains the conjugate points, the convergence map, the (total) shear map as a vector field, and the magnification map
        # - the following rows are for each individual plane, each one showing, in order:
        #   - the data with the light in that plane subtracted
        #   - that light model in the image plane
        #   - that light model in its own plane
        #   - that light model in its own plane with caustics overlayed

        # Create the figure and axes following the same dimensions as the model_summary function
        n_cols = 4
        n_rows = 2 + num_planes
        figsize = (15, n_rows*5)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Get the data and model images
        if hasattr(self, '_data'):
            data = self._data
        else:
            data = np.zeros(lens_image.Grid.number_pixel_axes)

        # Get the model image
        model = lens_image.model(
            **kwargs_result, 
        )

        # Get the residuals
        model_residuals, residuals = lens_image.normalized_residuals(
            data, model, kwargs_noise=kwargs_noise, mask=likelihood_mask,
        )
        if masked_residuals is True:
            residuals_plot = model_residuals
        else:
            residuals_plot = residuals
        red_chi2 = lens_image.reduced_chi2(
            data, model, kwargs_noise=kwargs_noise, mask=likelihood_mask,
        )
        # Get the extent of the image
        extent = lens_image.Grid.plt_extent
        # Get the critical lines and caustics
        clines_per_plane = []
        caustics_per_plane = []
        centers_per_plane = []
        for idx_plane in range(1, num_planes)[::-1]:
            clines, caustics, centers = model_util.critical_lines_caustics(
                lens_image, 
                kwargs_result['kwargs_mass'], 
                eta_flat=kwargs_result['eta_flat'], 
                return_lens_centers=True,
                k_plane=idx_plane,
            )
            clines_per_plane.append(clines)
            caustics_per_plane.append(caustics)
            centers_per_plane.append(centers)
        # Get the convergence map
        # x_grid, y_grid = lens_image.Grid.pixel_coordinates
        x_grid_highres, y_grid_highres = lens_image.Grid.create_model_grid(
            pixel_scale_factor=0.2
        ).pixel_coordinates
        kappa = lens_image.MPMassModel.kappa(
            x_grid_highres, y_grid_highres, 
            kwargs_result['eta_flat'],
            kwargs_result['kwargs_mass'], 
        )
        kappa = kappa[-1]
        # Get the magnification map
        magnification = lens_image.MPMassModel.magnification(
            x_grid_highres, y_grid_highres, 
            kwargs_result['eta_flat'],
            kwargs_result['kwargs_mass'], 
        )
        magnification = magnification[-1]
        # Get the shear field
        if show_shear_field:
            shear_field = model_util.total_shear_deflection_field(
                lens_image, 
                kwargs_result['kwargs_mass'], 
                eta_flat=kwargs_result['eta_flat'],
                num_pixels=20,
                k_plane=-1,
            )
        else:
            shear_field = None
        # Get the conjugate points
        conj_points_per_plane = lens_image.conjugate_points
        no_conj_points = all([cp is None for cp in conj_points_per_plane])
        traced_conj_points_per_plane = []
        for idx_plane in range(1, num_planes):
            traced_conj_points = lens_image.trace_conjugate_points(
                kwargs_result['eta_flat'],
                kwargs_result['kwargs_mass'], 
                N=idx_plane, k_mass=None,
            )
            traced_conj_points_per_plane.append(traced_conj_points)
        # print("conjugate points:", conj_points_per_plane)
        # print("traced conjugate points:", traced_conj_points_per_plane)

        # We also get the convergence and lens light in plane 0 to compare their isocontours
        # Get the lens light model
        if not lens_image.MPLightModel.light_models[0].has_pixels:
            main_lens_light = lens_image.MPLightModel.light_models[0].surface_brightness( 
                x_grid_highres, y_grid_highres, 
                kwargs_result['kwargs_light'][0], 
                pixels_x_coord=None, pixels_y_coord=None,  # NOTE: this means it would not work with a pixelated ,
            )
        else:
            # TODO: this has not been tested
            main_lens_light = lens_image.model(
                k_light=[[0]],
                **kwargs_result,
            )
            main_lens_light = main_lens_light[0]
        main_lens_mass = lens_image.MPMassModel.mass_models[0].kappa(
            x_grid_highres, y_grid_highres, 
            kwargs_result['kwargs_mass'][0], 
        )
        
        # Populate the figure
        # Data
        ax = axes[0, 0]
        im = ax.imshow(data, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
        im.set_rasterized(True)
        ax.set_title("Data", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        
        # Model
        ax = axes[0, 1]
        im = ax.imshow(model, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
        im.set_rasterized(True)
        self._plot_masks_contours(
            ax, lens_image, likelihood_mask=likelihood_mask, extent=extent, color='white', alpha=0.4,
        )
        ax.set_title("Model", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        
        # Residuals
        ax = axes[0, 2]
        im = ax.imshow(residuals_plot, cmap=self.cmap_res, extent=extent, norm=self.norm_res)
        im.set_rasterized(True)
        self._plot_masks_contours(
            ax, lens_image, likelihood_mask=likelihood_mask, extent=extent, color='black', alpha=0.6,
        )
        if likelihood_mask is not None and masked_residuals is False:
            ax.contour(likelihood_mask, extent=extent, levels=[0], 
                       colors='black', alpha=0.5, linewidths=0.5)
            
        ax.set_title(r"(f${}_{\rm data}$ - f${}_{\rm model})/\sigma$", fontsize=self.base_fontsize)
        nice_colorbar_residuals(im, residuals_plot, position='top', pad=0.4, size=0.2, 
                                vmin=self.norm_res.vmin, vmax=self.norm_res.vmax,
                                colorbar_kwargs={'orientation': 'horizontal'})
        text = r"$\chi^2_\nu={:.2f}$".format(red_chi2)
        ax.text(0.05, 0.05, text, color='black', fontsize=self.base_fontsize-4, 
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8})
        
        # Model with critical lines overlayed
        ax = axes[0, 3]
        im = ax.imshow(data, extent=extent, cmap=self.cmap_bw, norm=self.norm_flux)
        im.set_rasterized(True)
        ax.set_title("Data + critical lines", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        if likelihood_mask is not None:
            ax.contour(likelihood_mask, extent=extent, levels=[0], 
                       colors='black', alpha=0.5, linewidths=0.5)
        for i, clines in enumerate(clines_per_plane[::-1]):
            src_plane_idx = i + 1
            for j, curve in enumerate(clines):
                ax.plot(curve[0], curve[1], linewidth=2, color=colors_planes[src_plane_idx], linestyle=linestyles_planes[src_plane_idx], 
                        label=f"Source in plane {src_plane_idx}" if j == 0 else None)
        ax.scatter(*centers, s=20, c='gray', marker='+', linewidths=0.5)
        ax.legend()
        # tighten the layout and show the figure
        fig.tight_layout()

        # Conjugate points
        ax = axes[1, 0]
        if not no_conj_points:
            self._plot_conjugate_points(
                ax, data, num_planes, 
                conj_points_per_plane, 
                traced_conj_points_per_plane,
                extent, extent_zoom=None,
                colors_planes=colors_planes[1:],
            )
            ax.set_title("Data + conjugate points", fontsize=self.base_fontsize)
        else:
            ax.axis('off')
            # ax.set_title("No conjugate points", fontsize=self.base_fontsize)
        
        # Zoom-in on the traced conjuate points, or the shear field
        ax = axes[1, 1]
        if show_shear_field:
            x_field, y_field, gx_field, gy_field = shear_field
            ax.quiver(
                x_field, y_field,
                gx_field, gy_field, 
                scale=3, 
                width=0.1,
                scale_units='xy', 
                units='xy',
                pivot='middle',
                headaxislength=0, headlength=0,
                color='black', alpha=0.3,
            )
            ax.set_aspect('equal')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_title(f"Shear field (w.r.t. plane {num_planes-1})", fontsize=self.base_fontsize)
        else:
            self._plot_conjugate_points(
                ax, data, num_planes, 
                conj_points_per_plane, traced_conj_points_per_plane,
                extent, extent_zoom=extent_zoom,
                colors_planes=colors_planes[1:],
            )

        # Convergence map
        ax = axes[1, 2]
        im = ax.imshow(kappa, extent=extent, cmap=self.cmap_default, norm=norm_kappa)
        im.set_rasterized(True)
        ax.set_title(r"Convergence $\kappa$ (image plane)", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        # overlay the contours of the lens light and mass models
        contour_with_legend(
            ax,
            [
                main_lens_mass, 
                main_lens_light,
            ], 
            [
                dict(extent=extent, levels=10,
                     colors='black', alpha=0.5,
                     linewidths=0.5, linestyles='-',
                     label="Main lens mass"),
                dict(extent=extent, levels=5,
                     colors='black', alpha=0.5,
                     linewidths=1, linestyles='--',
                     label="Main lens light"),
            ], 
            logscale=True,
        )

        # Magnification map
        ax = axes[1, 3]
        im = ax.imshow(magnification, extent=extent, cmap=self.cmap_default, norm=Normalize(-20, 20), alpha=0.7)
        im.set_rasterized(True)
        ax.set_title(r"Magnification $\mu$ " + f"(w.r.t. plane {num_planes-1})", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        
        # The remaining axes are the same for each single plane that contains light (lens or source) models
        for i, idx_plane in enumerate(range(num_planes)):
            i_row = i + 2
            axes_row = axes[i_row, :]
            kwargs_grid_source_i = kwargs_grid_source[i-1] if i > 1 else None  # skip lens light plane
            self._plot_data_with_single_plane_model(
                axes_row,
                idx_plane,
                data,
                lens_image,
                kwargs_result,
                extent,
                kwargs_grid_source=kwargs_grid_source_i,
                linestyle=linestyles_planes[i],
                color=colors_planes[i],
            )

        fig.tight_layout()
        return fig



    def imshow_flux(self, ax, image, colorbar=True):
        im = ax.imshow(image, cmap=self.cmap_flux, norm=self.norm_flux)
        if colorbar is True:
            nice_colorbar(im)


    def _get_norm_for_model(self, model, lock_colorbars):
        if lock_colorbars is True and self.norm_flux is None:
            if self.flux_log_scale is True:
                norm_flux = LogNorm(model.min(), model.max())
            else:
                norm_flux = Normalize(model.min(), model.max())
        else:
            norm_flux = self.norm_flux
        return norm_flux
    
    @staticmethod
    def _plot_masks_contours(ax, lens_image, likelihood_mask=None, extent=None, alpha=0.5, color='red'):
        if likelihood_mask is not None:
            ax.contour(likelihood_mask, extent=extent, levels=[0], 
                    colors=color, alpha=alpha, linewidths=1.5, linestyles='-',
                    label="Likelhood mask")
        if not all([m is None for m in lens_image.source_arc_masks]):
            for j, mask in enumerate(lens_image.source_arc_masks):
                ax.contour(mask, extent=extent, levels=[0], 
                        colors=color, alpha=alpha, linewidths=1.5, linestyles=':',
                        label="Source arc mask" if j == 0 else None)


    def _plot_conjugate_points(
            self, ax, data, num_planes, 
            conj_points_per_plane, traced_conj_points_per_plane, 
            extent, extent_zoom=None,
            colors_planes=['tab:orange', 'tab:purple', 'tab:cyan', 'tab:pink', 'tab:blue', 'tab:red'],
        ):
        im = ax.imshow(data, extent=extent, cmap=self.cmap_bw, norm=self.norm_flux, 
                       alpha=1 if extent_zoom is None else 0)
        for idx_plane in range(num_planes-1):
            if conj_points_per_plane[idx_plane] is None:
                continue
            color = colors_planes[idx_plane]
            ax.scatter(*conj_points_per_plane[idx_plane].T, s=140, edgecolors=color, marker='o', linewidths=2, 
                    facecolors='none')
            ax.scatter(*traced_conj_points_per_plane[idx_plane].T, s=400 if extent_zoom else 100, c=color, marker='*', linewidths=1)
            for conj_point, traced_conj_point in zip(conj_points_per_plane[idx_plane], traced_conj_points_per_plane[idx_plane]):
                conj_point_arrow = FancyArrowPatch(
                    (traced_conj_point[0], traced_conj_point[1]), 
                    (conj_point[0], conj_point[1]), 
                    arrowstyle=ArrowStyle("Fancy", head_length=1, head_width=1, tail_width=1), 
                    color=color, lw=2, alpha=0.2, clip_on=True,
                )
                ax.add_patch(conj_point_arrow)
        # ax.legend()
        ax.set_aspect('equal')
        lim_extent = extent_zoom if extent_zoom is not None else extent
        ax.set_xlim(lim_extent[0], lim_extent[1])
        ax.set_ylim(lim_extent[2], lim_extent[3])
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'},
                      invisible=True)
        
    def _plot_data_with_single_plane_model(
        self,
        axes_row,
        k_plane,
        data,
        lens_image,
        kwargs_result,
        extent,
        kwargs_grid_source=None,
        linestyle='-',
        color='tab:blue',
        ):
        """Populate axes (single row) with, in order:
           - the data with the light in that plane subtracted
           - that light model in the image plane
           - that light model in its own plane
           - that light model in its own plane with caustics overlayed
        """
        # Get the lensed source model of the required plane
        observed_light_model = lens_image.model(
            k_planes=k_plane,
            **kwargs_result,
        )
        # Subtract the lensed source model from the data
        data_subtracted = data - observed_light_model
        # Get the unlensed source model of the required plane
        if k_plane > 0:
            plane_has_pixels = lens_image.MPLightModel.light_models[k_plane].has_pixels
            if plane_has_pixels:
                # TODO: the following complicated code should be implemented wthin MPLensImage (similar to LensImage)
                # get the adapted coordinates axes
                x_coord, y_coord, orig_extent_tmp = lens_image.get_source_coordinates(
                    kwargs_result['eta_flat'],
                    kwargs_result['kwargs_mass'],
                )
                # select the right plane
                x_coord, y_coord = x_coord[k_plane], y_coord[k_plane]
                orig_extent_tmp = orig_extent_tmp[k_plane]  # NOTE: this is not a "plot" extent (misses half pixels at each end)
            else:
                x_coord, y_coord = None, None
            if kwargs_grid_source is not None:
                grid_src = lens_image.Grid.create_model_grid(**kwargs_grid_source)
                x_grid_src, y_grid_src = grid_src.pixel_coordinates
                orig_extent = grid_src.plt_extent
            elif plane_has_pixels:
                # create a 2d grid out of the axes
                x_grid_src, y_grid_src = np.meshgrid(x_coord, y_coord)
                # transform the proper extent to the one usable with imshow 
                pix_scl_x = jnp.abs(orig_extent_tmp[0]-orig_extent_tmp[1])
                pix_scl_y = jnp.abs(orig_extent_tmp[2]-orig_extent_tmp[3])
                half_pix_scl = jnp.sqrt(pix_scl_x*pix_scl_y) / 2.
                orig_extent = [
                    orig_extent_tmp[0]-half_pix_scl, orig_extent_tmp[1]+half_pix_scl, 
                    orig_extent_tmp[2]-half_pix_scl, orig_extent_tmp[3]+half_pix_scl
                ]
            else:
                # fall-back case when no grid is provided nor found
                x_grid_src, y_grid_src = lens_image.ImageNumerics.coordinates_evaluate  # NOTE: coordinates are 1d here
                orig_extent = extent
            original_light_model = lens_image.MPLightModel.light_models[k_plane].surface_brightness(
                x_grid_src, y_grid_src, 
                kwargs_result['kwargs_light'][k_plane],
                pixels_x_coord=x_coord,
                pixels_y_coord=y_coord,
            ) * lens_image.Grid.pixel_area
            # if original_light_model.ndim == 1:
            #     original_light_model = lens_image.ImageNumerics.re_size_convolve(
            #         original_light_model, unconvolved=True, input_as_list=False,
            #     )
        else:
            # for the lens plane, we do not have a source model
            # original_light_model = lens_image.model(
            #     k_planes=k_plane,
            #     unconvolved=True,
            #     **kwargs_result,
            # )
            # orig_extent = extent
            x_grid_model, y_grid_model = lens_image.ImageNumerics.coordinates_evaluate  # NOTE: coordinates are 1d here
            original_light_model = lens_image.MPLightModel.light_models[0].surface_brightness(
                x_grid_model, y_grid_model, 
                kwargs_result['kwargs_light'][0],
                pixels_x_coord=None,
                pixels_y_coord=None,
            ) * lens_image.Grid.pixel_area
            orig_extent = extent
        
        # typically when we used lens_image.ImageNumerics.coordinates_evaluate
        if original_light_model.ndim == 1:
            n_side = int(np.sqrt(original_light_model.size))
            original_light_model = original_light_model.reshape((n_side, n_side))

        # Get the caustics
        _, caustics = model_util.critical_lines_caustics(
            lens_image, 
            kwargs_result['kwargs_mass'], 
            eta_flat=kwargs_result['eta_flat'], 
            return_lens_centers=False,
            k_plane=k_plane,
            supersampling=6,
        )

        # Get the traced conjugate points
        traced_conj_points = lens_image.trace_conjugate_points(
            kwargs_result['eta_flat'],
            kwargs_result['kwargs_mass'], 
            N=k_plane,
        )

        # Source-subtracted data
        ax = axes_row[0]
        im = ax.imshow(data_subtracted, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
        im.set_rasterized(True)
        ax.set_title(f"Subtracted model in plane {k_plane}", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        
        # Light model in image plane
        ax = axes_row[1]
        im = ax.imshow(observed_light_model, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
        im.set_rasterized(True)
        ax.set_title(f"Obs. light model in plane {k_plane}", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                      colorbar_kwargs={'orientation': 'horizontal'})
        
        # Light model in "original form"
        ax = axes_row[2]
        im = ax.imshow(original_light_model, extent=orig_extent, cmap=self.cmap_flux_alt, norm=self.norm_flux)
        im.set_rasterized(True)
        if k_plane > 0:
            ax.set_title(f"Unlensed model in plane {k_plane}", fontsize=self.base_fontsize)
        else:
            ax.set_title(f"Unconvolved model in plane {k_plane}", fontsize=self.base_fontsize)
        nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                        colorbar_kwargs={'orientation': 'horizontal'})
            
        # Source model with caustics overlayed )
        ax = axes_row[3]
        if k_plane > 0:
            im = ax.imshow(original_light_model, extent=orig_extent, cmap=self.cmap_bw, norm=self.norm_flux)
            im.set_rasterized(True)
            nice_colorbar(im, position='top', pad=0.4, size=0.2, 
                          colorbar_kwargs={'orientation': 'horizontal'})
            ax.set_title(f"With caustics + conj. points", fontsize=self.base_fontsize)
            # FIXME: caustics are not plotted correctly
            for j, curve in enumerate(caustics):
                # print(j, curve)
                ax.plot(curve[0], curve[1], linewidth=2, color=color, linestyle=linestyle)
            ax.scatter(*traced_conj_points.T, c=color, s=300, marker='*', linewidths=1, edgecolors='white', facecolors=color)
            ax.set_xlim(orig_extent[0], orig_extent[1])
            ax.set_ylim(orig_extent[2], orig_extent[3])
            ax.set_aspect('equal')
        else:
            ax.axis('off')
