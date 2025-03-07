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

from herculens.Util.plot_util import nice_colorbar, nice_colorbar_residuals
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
            k_source=None, k_lens=None,
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
            model = lens_image.model(
                **kwargs_result, 
                k_lens=k_lens,
                adapted_source_pixels_coords=adapted_source_pixels_coords,
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
                                    markeredgecolor='black', markersize=10, marker='o', 
                                    fillstyle='none', markeredgewidth=0.5)
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
