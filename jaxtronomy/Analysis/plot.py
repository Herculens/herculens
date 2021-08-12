import copy
import warnings
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm

from jaxtronomy.Util.plot_util import nice_colorbar, nice_colorbar_residuals
from jaxtronomy.Util import image_util

# Some general default for plotting
plt.rc('image', interpolation='none', origin='lower')  # imshow


class Plotter(object):

    # Define some custom colormaps
    try:
        import palettable
    except ImportError:
        cmap_base = plt.get_cmap('cubehelix')
    else:
        cmap_base = palettable.cubehelix.Cubehelix.make(name='flux_colormap',
                                                        start=0.5,
                                                        rotation=-1,
                                                        gamma=0.8,
                                                        sat=0.8,
                                                        n=256).mpl_colormap
    cmap_base.set_under('black')
    cmap_base.set_over('white')
    cmap_flux = copy.copy(cmap_base)
    cmap_flux.set_bad(color='black')
    cmap_flux_alt = copy.copy(cmap_base)
    cmap_flux_alt.set_bad(color='#222222')  # emphasize e.g. non-positive pixels in log scale
    cmap_resid = plt.get_cmap('RdBu_r')
    cmap_default = plt.get_cmap('viridis')
    cmap_deriv1 = plt.get_cmap('cividis')
    cmap_deriv2 = plt.get_cmap('inferno')

    def __init__(self, base_fontsize=18, flux_log_scale=True, 
                 flux_vmin=None, flux_vmax=None):
        self._base_fs = base_fontsize
        if flux_log_scale is True:
            self.norm_flux = LogNorm(flux_vmin, flux_vmax)
        else:
            self.norm_flux = None

    def set_data(self, data):
        self._data = data

    def set_true_source(self, true_source):
        self._true_source = true_source

    def set_true_potential_perturbations(self, true_potential):
        self._true_pot_perturb = true_potential

    def model_summary(self, lens_image, kwargs_result,
                      show_image=True, show_source=True, show_lens_mass=False,
                      reproject_pixelated_models=False, shift_potential_model='min',
                      likelihood_mask=None, potential_mask=None):
        n_cols = 3
        n_rows = sum([show_image, show_source, show_lens_mass, show_lens_mass])
        
        extent = lens_image.Grid.extent
        if lens_image.SourceModel.has_pixels:
            src_extent = lens_image.Grid.model_pixel_extent('source')
        else:
            src_extent = extent
            
        if show_image:
            if hasattr(self, '_data'):
                data = self._data
            else:
                data = np.zeros_like(model)

            # create the resulting model image
            model = lens_image.model(**kwargs_result)
            noise_var = lens_image.Noise.C_D
            if likelihood_mask is None:
                likelihood_mask = np.ones_like(model)

        if show_source:
            kwargs_source = copy.deepcopy(kwargs_result['kwargs_source'])
            if lens_image.SourceModel.has_pixels:
                src_idx = lens_image.SourceModel.pixelated_index
                if reproject_pixelated_models:
                    # we need to make sure it's jax.numpy array for source_surface_brightness when using PIXELATED source profile
                    kwargs_source[src_idx]['pixels'] = jnp.asarray(kwargs_source[src_idx]['pixels'])
                    # we extract the right coordinate arrays
                    x_grid_src, y_grid_src = lens_image.Grid.model_pixel_coordinates('source')
                    source_model = lens_image.SourceModel.surface_brightness(x_grid_src, y_grid_src, kwargs_source)
                    source_model *= lens_image.Grid.pixel_area
                else:
                    source_model = kwargs_source[src_idx]['pixels']
            else:
                source_model = lens_image.source_surface_brightness(kwargs_source, de_lensed=True, unconvolved=True)

            if hasattr(self, '_true_source'):
                true_source = self._true_source
            else:
                true_source = np.zeros_like(source_model)

            if source_model.size != true_source.size:
                npix_true = len(true_source)
                x_coords_true = np.linspace(extent[0], extent[1], npix_true)
                y_coords_true = np.linspace(extent[2], extent[3], npix_true)
                if lens_image.SourceModel.has_pixels:
                    x_coords_src, y_coords_src = lens_image.Grid.model_pixel_axes('source')
                else:
                    npix_src = len(source_model)
                    x_coords_src = np.linspace(extent[0], extent[1], npix_src)
                    y_coords_src = np.linspace(extent[2], extent[3], npix_src)
                true_source = image_util.re_size_array(x_coords_true, y_coords_true, true_source, x_coords_src, y_coords_src)
                warnings.warn("True source array has been interpolated to match model array")

        if show_lens_mass:
            # TODO: update those lines to be consistent with source model above
            pot_idx = -1  # here we assume the last lens profile is 'PIXELATED'
            x_coords_pot, y_coords_pot = lens_image.LensModel.pixelated_coordinates
            x_grid_lens, y_grid_lens = np.meshgrid(x_coords_pot, y_coords_pot)
            alpha_x, alpha_y = lens_image.LensModel.alpha(x_grid_lens, y_grid_lens, 
                                                          kwargs_result['kwargs_lens'], k=pot_idx)
            kappa = lens_image.LensModel.kappa(x_grid_lens, y_grid_lens, 
                                               kwargs_result['kwargs_lens'], k=pot_idx)
            potential_model = np.copy(kwargs_result['kwargs_lens'][pot_idx]['pixels'])
            #potential_model = lens_image.LensModel.potential(x_grid_lens, y_grid_lens,
            #                                                 kwargs_result['kwargs_lens'], k=pot_idx)

            # here we know that there are no perturbations in the true potential
            if hasattr(self, '_true_pot_perturb'):
                true_potential = self._true_pot_perturb
            else:
                true_potential = np.zeros_like(potential_model)
            
            if shift_potential_model == 'min':
                min_in_mask = (potential_model * potential_mask).min()
                potential_model = potential_model - min_in_mask
                print("delta_psi shift by min:", min_in_mask)
            elif shift_potential_model == 'mean':
                mean_in_mask = (potential_model * potential_mask).mean()
                true_mean_in_mask = (true_potential * potential_mask).mean()
                potential_model = potential_model - mean_in_mask + true_mean_in_mask
                print("delta_psi shift by mean values:", mean_in_mask, true_mean_in_mask)

            if potential_mask is None:
                # TODO: compute potential mask based on undersampled likelihood_mask
                potential_mask = np.ones_like(potential_model)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*5))
        if len(axes.shape) == 1:
            axes.reshape((n_rows, n_cols))
        i_row = 0

        if show_image:

            ##### IMAGING DATA AND MODEL IMAGE #####
            ax = axes[i_row, 0]
            im = ax.imshow(data * likelihood_mask, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
            ax.set_title("Data", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 1]
            im = ax.imshow(model, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
            ax.set_title("Model", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 2]
            norm_res = lens_image.normalized_residuals(data, model, mask=likelihood_mask)
            red_chi2 = lens_image.reduced_chi2(data, model, mask=likelihood_mask)
            im = ax.imshow(norm_res * likelihood_mask, cmap=self.cmap_resid, vmin=-4, vmax=4, extent=extent)
            ax.set_title("Norm. residuals ("+r"$\chi^2$"+f"={red_chi2:.2f})", fontsize=self._base_fs)
            nice_colorbar_residuals(im, norm_res, vmin=-4, vmax=4)
            i_row += 1

        if show_source:

            ##### UNLENSED AND UNCONVOLVED SOURCE MODEL #####
            ax = axes[i_row, 0]
            im = ax.imshow(true_source, extent=src_extent, cmap=self.cmap_flux_alt, norm=self.norm_flux) #, vmax=vmax)
            nice_colorbar(im)
            ax.set_title("True source", fontsize=self._base_fs)
            ax = axes[i_row, 1]
            im = ax.imshow(source_model, extent=src_extent, cmap=self.cmap_flux_alt, norm=self.norm_flux) #, vmax=vmax)
            #im = ax.imshow(source_model, extent=extent, cmap=self.cmap_flux_alt, norm=LogNorm(1e-5))
            nice_colorbar(im)
            ax.set_title("Source model", fontsize=self._base_fs)
            ax = axes[i_row, 2]
            diff = true_source - source_model
            vmax_diff = true_source.max() / 10.
            im = ax.imshow(diff, extent=src_extent, 
                           cmap=self.cmap_resid, vmin=-vmax_diff, vmax=vmax_diff)
            ax.set_title("Residuals", fontsize=self._base_fs)
            nice_colorbar_residuals(im, diff, vmin=-vmax_diff, vmax=vmax_diff)
            i_row += 1

        if show_lens_mass:

            ##### PIXELATED POTENTIAL PERTURBATIONS #####
            ax = axes[i_row, 0]
            im = ax.imshow(true_potential * potential_mask, cmap=self.cmap_default, extent=extent)
            ax.set_title("True $\delta\psi$", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 1]
            im = ax.imshow(potential_model * potential_mask, cmap=self.cmap_default, extent=extent)
            ax.set_title("$\delta\psi$ model", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 2]
            pot_abs_res = (true_potential - potential_model) * potential_mask
            vmax = np.max(np.abs(true_potential)) / 2.
            im = ax.imshow(pot_abs_res, cmap=self.cmap_resid, vmin=-vmax, vmax=vmax, extent=extent)
            ax.set_title("Residuals", fontsize=self._base_fs)
            nice_colorbar_residuals(im, pot_abs_res, vmin=-vmax, vmax=vmax)
            i_row += 1

            ##### DEFLECTION ANGLES AND SURFACE MASS DENSITY #####
            ax = axes[i_row, 0]
            ax.set_title(r"$\delta\alpha_x$ model", fontsize=self._base_fs)
            im = ax.imshow(alpha_x * potential_mask, cmap=self.cmap_deriv1, alpha=1, extent=extent)
            nice_colorbar(im)
            ax = axes[i_row, 1]
            alpha_y_show = alpha_y
            ax.set_title(r"$\delta\alpha_y$ model", fontsize=self._base_fs)
            im = ax.imshow(alpha_y * potential_mask, cmap=self.cmap_deriv1, alpha=1, extent=extent)
            nice_colorbar(im)
            ax = axes[i_row, 2]
            kappa_smoothed = ndimage.gaussian_filter(kappa, 1)
            ax.set_title(r"$\delta\kappa$ model (smoothed)", fontsize=self._base_fs)
            im = ax.imshow(kappa_smoothed * potential_mask, cmap=self.cmap_deriv2, alpha=1, extent=extent) #, vmin=0)
            nice_colorbar(im)
        
        plt.show()