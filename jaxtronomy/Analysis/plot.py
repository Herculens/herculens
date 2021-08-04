import copy
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm

from jaxtronomy.Util.plot_util import nice_colorbar, nice_colorbar_residuals

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

    def model_summary(self, data, lens_image, kwargs_result, true_source=None, true_potential=None,
                      show_image=True, show_source=True, show_lens_mass=False,
                      shift_potential='min', with_mask=False,
                      data_mask=None, potential_mask=None):
        if data_mask is not None:
            raise NotImplementedError("Data mask not yet supported")

        n_cols = 3
        n_rows = sum([show_image, show_source, show_lens_mass, show_lens_mass])
        
        x_grid, y_grid = lens_image.Grid.pixel_coordinates
        x_coords = x_grid[0, :]
        y_coords = y_grid[:, 0]
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        if show_image:
            # create the resulting model image
            model = lens_image.model(**kwargs_result)
            noise_var = lens_image.Noise.C_D

        if show_source:
            kwargs_source = copy.deepcopy(kwargs_result['kwargs_source'])
            if 'image' in kwargs_source[0]:
                # we need to make sure it's jax.numpy array for source_surface_brightness when using PIXELATED source profile
                kwargs_source[0]['image'] = jnp.asarray(kwargs_source[0]['image'])
            source_model = lens_image.source_surface_brightness(kwargs_source, de_lensed=True, unconvolved=True)
            if true_source is None:
                true_source = np.zeros_like(source_model)

        if show_lens_mass:
            # TODO: check that there is indeed a pixelated potential profile in the model
            pot_idx = -1  # here we assume the last lens profile is 'PIXELATED'
            x_coords_pot, y_coords_pot = lens_image.LensModel.pixelated_coordinates
            x_grid_lens, y_grid_lens = np.meshgrid(x_coords_pot, y_coords_pot)
            alpha_x, alpha_y = lens_image.LensModel.alpha(x_grid_lens, y_grid_lens, 
                                                          kwargs_result['kwargs_lens'], k=pot_idx)
            kappa = lens_image.LensModel.kappa(x_grid_lens, y_grid_lens, 
                                               kwargs_result['kwargs_lens'], k=pot_idx)
            # potential_model = np.copy(kwargs_result['kwargs_lens'][pot_idx]['psi_grid'])
            potential_model = lens_image.LensModel.potential(x_grid_lens, y_grid_lens,
                                                             kwargs_result['kwargs_lens'], k=pot_idx)

            # here we know that there are no perturbations in the true potential
            if true_potential is None:
                true_potential = np.zeros_like(potential_model)
            if shift_potential == 'min':
                min_in_mask = (potential_model * potential_mask).min()
                potential_model = potential_model - min_in_mask
                print("delta_psi shift by min:", min_in_mask)
            elif shift_potential == 'mean':
                mean_in_mask = (potential_model * potential_mask).mean()
                true_mean_in_mask = (true_potential * potential_mask).mean()
                potential_model = potential_model - mean_in_mask + true_mean_in_mask
                print("delta_psi shift & normalization by mean:", mean_in_mask, true_mean_in_mask)
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*5))
        if len(axes.shape) == 1:
            axes.reshape((n_rows, n_cols))

        i_row = 0

        if show_image:

            ##### IMAGING DATA AND MODEL IMAGE #####
            ax = axes[i_row, 0]
            im = ax.imshow(data, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
            ax.set_title("Data", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 1]
            im = ax.imshow(model, extent=extent, cmap=self.cmap_flux, norm=self.norm_flux)
            ax.set_title("Model", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 2]
            norm_res = (data - model) / np.sqrt(noise_var)
            red_chi2 = np.mean(norm_res**2)
            im = ax.imshow(norm_res, cmap=self.cmap_resid, vmin=-4, vmax=4, extent=extent)
            ax.set_title("Normalised residuals ("+r"$\chi^2$"+f"={red_chi2:.2f})", fontsize=self._base_fs)
            nice_colorbar_residuals(im, norm_res, vmin=-4, vmax=4)
            i_row += 1

        if show_source:

            ##### UNLENSED AND UNCONVOLVED SOURCE MODEL #####
            ax = axes[i_row, 0]
            im = ax.imshow(true_source, extent=extent, cmap=self.cmap_flux_alt, norm=self.norm_flux) #, vmax=vmax)
            nice_colorbar(im)
            ax.set_title("True source", fontsize=self._base_fs)
            ax = axes[i_row, 1]
            im = ax.imshow(source_model, extent=extent, cmap=self.cmap_flux_alt, norm=self.norm_flux) #, vmax=vmax)
            nice_colorbar(im)
            ax.set_title("Source model", fontsize=self._base_fs)
            ax = axes[i_row, 2]
            diff = true_source - source_model
            vmax_diff = true_source.max() / 10.
            im = ax.imshow(diff, extent=extent, 
                           cmap=self.cmap_resid, vmin=-vmax_diff, vmax=vmax_diff)
            ax.set_title("Residuals", fontsize=self._base_fs)
            nice_colorbar_residuals(im, diff, vmin=-vmax_diff, vmax=vmax_diff)
            i_row += 1

        if show_lens_mass:

            ##### PIXELATED POTENTIAL PERTURBATIONS #####
            ax = axes[i_row, 0]
            true_pot_show = true_potential
            if with_mask:
                true_pot_show *= potential_mask
            im = ax.imshow(true_pot_show, cmap=self.cmap_default, extent=extent)
            ax.set_title("True $\delta\psi$", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 1]
            pot_model_show = potential_model
            if with_mask:
                pot_model_show *= potential_mask
            im = ax.imshow(pot_model_show, cmap=self.cmap_default, extent=extent)
            ax.set_title("$\delta\psi$ model", fontsize=self._base_fs)
            nice_colorbar(im)
            ax = axes[i_row, 2]
            pot_abs_res_show = (true_potential - potential_model)
            if with_mask:
                pot_abs_res_show *= potential_mask
            vmax = np.max(np.abs(true_potential)) / 2.
            im = ax.imshow(pot_abs_res_show, cmap=self.cmap_resid, vmin=-vmax, vmax=vmax, extent=extent)
            ax.set_title("Residuals", fontsize=self._base_fs)
            nice_colorbar_residuals(im, pot_abs_res_show, vmin=-vmax, vmax=vmax)
            i_row += 1

            ##### DEFLECTION ANGLES AND SURFACE MASS DENSITY #####
            ax = axes[i_row, 0]
            alpha_x_show = alpha_x
            if with_mask:
                alpha_x_show *= potential_mask
            ax.set_title(r"$\delta\alpha_x$ model", fontsize=self._base_fs)
            im = ax.imshow(alpha_x_show, cmap=self.cmap_deriv1, alpha=1, extent=extent)
            nice_colorbar(im)
            ax = axes[i_row, 1]
            alpha_y_show = alpha_y
            if with_mask:
                alpha_y_show *= potential_mask
            ax.set_title(r"$\delta\alpha_y$ model", fontsize=self._base_fs)
            im = ax.imshow(alpha_y_show, cmap=self.cmap_deriv1, alpha=1, extent=extent)
            nice_colorbar(im)
            ax = axes[i_row, 2]
            #kappa_show = kappa
            kappa_show = ndimage.gaussian_filter(kappa, 1)
            if with_mask:
                kappa_show *= potential_mask
            ax.set_title(r"$\delta\kappa$ model (smoothed)", fontsize=self._base_fs)
            im = ax.imshow(kappa_show, cmap=self.cmap_deriv2, alpha=1, extent=extent) #, vmin=0)
            nice_colorbar(im)
        
        plt.show()