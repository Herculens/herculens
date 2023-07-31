# Defines the model of a strong lens
# 
# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2020, Austin Peel and Aymeric Galan
# based on the LensingOperator class from slitronomy (https://github.com/aymgal/SLITronomy)

__author__ = 'austinpeel', 'aymgal'


import copy
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax.experimental import sparse as jsparse

from herculens.Util import util


__all__ = ['LensingOperator', 'PlaneGrid', 'MaskedPlaneGrid']


class LensingOperator(object):

    """Defines the mapping of pixelated light profiles between image and source planes"""

    def __init__(self, mass_model_class, image_grid_class, source_grid_class, arc_mask=None):
        """Summary
        
        Parameters
        ----------
        mass_model_class : TYPE
            Description
        image_grid_class : TYPE
            Description
        source_grid_class : TYPE
            Description
        
        Raises
        ------
        ValueError
            Description
        """
        self.MassModel = mass_model_class
        self.ImagePlane = MaskedPlaneGrid(image_grid_class, mask=arc_mask)
        self.SourcePlane = PlaneGrid(source_grid_class)

    @partial(jit, static_argnums=(0,))
    def source2image(self, source_1d):
        image = self._mapping @ source_1d
        return self.ImagePlane.place(image)

    @partial(jit, static_argnums=(0,))
    def source2image_2d(self, source):
        source_1d = util.image2array(source)
        return util.array2image(self.source2image(source_1d))

    @partial(jit, static_argnums=(0, 2))
    def image2source(self, image_1d, no_flux_norm=False):
        """if no_flux_norm is True, do not normalize light flux to better visualize the mapping"""
        source = self._mapping.T @ self.ImagePlane.extract(image_1d)
        if not no_flux_norm:
            source /= self._norm_image2source
        return source

    @partial(jit, static_argnums=(0, 2))
    def image2source_2d(self, image, no_flux_norm=False):
        image_1d = util.image2array(image)
        return util.array2image(self.image2source(image_1d, no_flux_norm=no_flux_norm))

    @partial(jit, static_argnums=(0,))
    def lensing(self, source):
        return self.source2image_2d(source)

    @partial(jit, static_argnums=(0,))
    def lensing_transpose(self, image):
        return self.image2source_2d(image)

    @property
    def source_plane_coordinates(self):
        return self.SourcePlane.theta_x, self.SourcePlane.theta_y

    @property
    def image_plane_coordinates(self):
        return self.ImagePlane.theta_x, self.ImagePlane.theta_y

    @property
    def pixel_area_ratio(self):
        """source pixel area divide by image pixel area"""
        return (self.SourcePlane.delta_pix / self.ImagePlane.delta_pix)**2

    def get_lens_mapping(self):
        if not hasattr(self, '_mapping'):
            return None
        return self._mapping, self._norm_image2source

    def delete_cache(self):
        if hasattr(self, '_mapping'):
            del self._mapping
        if hasattr(self, '_norm_image2source'):
            del self._norm_image2source

    def compute_mapping(self, kwargs_lens):
        # delete cached mapping matrices
        self.delete_cache()

        # compute mapping between image and source planes due to lensing, on original source plane grid
        self._mapping, self._norm_image2source = self._compute_mapping(kwargs_lens)

    def _compute_mapping(self, kwargs_lens):
        return self._compute_mapping_bilinear(kwargs_lens)

    def _compute_mapping_bilinear(self, kwargs_lens, resized_source_plane=True):
        """Compute the mapping between image and source plane pixels.

        This method ray-traces the image plane pixel
        coordinates back to the source plane and regularizes the resultig
        positions to a grid. In contrast to the 'nearest' interpolation scheme,
        this mapping incorporates a bilinear weighting to interpolate flux on
        the source plane following Treu & Koopmans (2004).

        """
        # Get image plane coordinates
        theta_x, theta_y = self.image_plane_coordinates

        # Compute lens mapping from image to source coordinates
        beta_x, beta_y = self.MassModel.ray_shooting(theta_x, theta_y, kwargs_lens)

        # Determine source pixels and their appropriate weights
        indices, weights = self._find_source_pixels_bilinear(beta_x, beta_y, 0., 0.)
    
        # Build lensing matrix as a sparse matrix for saving memory
        dense_shape = (self.ImagePlane.grid_size, self.SourcePlane.grid_size)
        lens_mapping = jsparse.BCOO((weights, np.stack(indices, axis=1)), 
                                    shape=dense_shape, indices_sorted=True)

        # Compue the flux normalization factors
        norm_image2source = np.maximum(1, lens_mapping.sum(axis=0).todense())
        return lens_mapping, norm_image2source

    def _find_source_pixels_bilinear(self, beta_x, beta_y, grid_offset_x, grid_offset_y, 
                                     warning=False):
        """Fast binning of ray-traced coordinates and weight calculation.

        Parameters
        ----------
        beta_x, beta_y : array-like
            Coordinates in the source plane of ray-traced points from the
            image plane.
        grid_offset_x, grid_offset_y : float
            Amount by which to shift the source plane grid in each direction.
        warning : bool
            Print a warning if any returned weights are negative.

        Returns
        -------
        (row, col), weight
            Weights are the source grid interpolation values, which belong
            at position (row, col) in the sparse lensing matrix. There are at
            most 4 weights corresponding to each row.

        Notes
        -----
        Ray-traced coordinates from the image plane are simply removed if they
        fall outside of the source plane grid, as is done in Treu & Koopmans
        (2004). Although this should only rarely occur in practice, e.g. for
        extreme parameters of the lens model, a better approach might still be
        to expand the source plane instead.

        """
        # Standardize inputs for vectorization
        beta_x = np.atleast_1d(beta_x)
        beta_y = np.atleast_1d(beta_y)
        assert len(beta_x) == len(beta_y), "Input arrays must be the same size."
        num_beta = len(beta_x)

        # Get source plane coordinates, shift them if necessary
        source_theta_x, source_theta_y = self.source_plane_coordinates
        source_theta_x += grid_offset_x
        source_theta_y += grid_offset_y

        # Compute bin edges so that (theta_x, theta_y) lie at the grid centers
        num_pix = self.SourcePlane.num_pix
        delta_pix = self.SourcePlane.delta_pix
        half_pix = delta_pix / 2

        theta_x = source_theta_x[:num_pix]
        x_dir = -1 if theta_x[0] > theta_x[-1] else 1  # Handle x-axis inversion
        # x_dir = lax.cond(theta_x[0] > theta_x[-1], lambda _: -1, lambda _: +1, None)  # Handle x-axis inversion
        x_lower = theta_x[0] - x_dir * half_pix
        x_upper = theta_x[-1] + x_dir * half_pix
        xbins = np.linspace(x_lower, x_upper, num_pix + 1)

        theta_y = source_theta_y[::num_pix]
        y_dir = -1 if theta_y[0] > theta_y[-1] else 1  # Handle y-axis inversion
        # y_dir = lax.cond(theta_y[0] > theta_y[-1], lambda _: -1, lambda _: +1, None)  # Handle y-axis inversion
        y_lower = theta_y[0] - y_dir * half_pix
        y_upper = theta_y[-1] + y_dir * half_pix
        ybins = np.linspace(y_lower, y_upper, num_pix + 1)

        # Keep only betas that fall within the source plane grid
        x_min, x_max = [x_lower, x_upper][::x_dir]
        y_min, y_max = [y_lower, y_upper][::y_dir]
        # x_min, x_max = lax.cond(x_dir == 1, lambda _: (x_lower, x_upper), lambda _: (x_upper, x_lower), None)
        # y_min, y_max = lax.cond(y_dir == 1, lambda _: (y_lower, y_upper), lambda _: (y_upper, y_lower), None)
        selection = ((beta_x > x_min) & (beta_x < x_max) &
                     (beta_y > y_min) & (beta_y < y_max))
        if np.any(1 - selection.astype(int)):
           beta_x = beta_x[selection]
           beta_y = beta_y[selection]
           num_beta = len(beta_x)

        # Find the (1D) source plane pixel that (beta_x, beta_y) falls in
        index_x = np.digitize(beta_x, xbins) - 1
        index_y = np.digitize(beta_y, ybins) - 1
        index_1 = index_x + index_y * num_pix

        # Compute distances between ray-traced betas and source grid points
        dx = beta_x - source_theta_x[index_1]
        dy = beta_y - source_theta_y[index_1]

        # Find the three other nearest pixels (may end up out of bounds)
        index_2 = index_1 + x_dir * np.sign(dx).astype(int)
        index_3 = index_1 + y_dir * np.sign(dy).astype(int) * num_pix
        index_4 = index_2 + y_dir * np.sign(dy).astype(int) * num_pix

        # Treat these index arrays as four sets stacked vertically
        # Prepare to mask out out-of-bounds pixels as well as repeats
        # The former is important for the csr_matrix to be generated correctly
        max_index = self.SourcePlane.grid_size - 1  # Upper index bound
        mask = np.ones((4, num_beta), dtype=bool)  # Mask for the betas

        # Mask out any neighboring pixels that end up out of bounds
        mask[1, np.where((index_2 < 0) | (index_2 > max_index))[0]] = False
        mask[2, np.where((index_3 < 0) | (index_3 > max_index))[0]] = False
        mask[3, np.where((index_4 < 0) | (index_4 > max_index))[0]] = False
        
        # Mask any repeated pixels (2 or 3x) arising from unlucky grid alignment
        zero_dx = np.where(dx == 0)[0]
        zero_dy = np.where(dy == 0)[0]
        unique, counts = np.unique(zero_dx + zero_dy, return_counts=True)
        repeat_row = [ii + 1 for c in counts for ii in range(0, 3, 3 - c)]
        repeat_col = [u for (u, c) in zip(unique, counts) for _ in range(c + 1)]
        mask[(repeat_row, repeat_col)] = False
        # mask.at[(repeat_row, repeat_col)].set(False)

        # Generate 2D indices of non-zero elements for the sparse matrix
        row = np.tile(np.nonzero(selection)[0], (4, 1))
        col = np.array([index_1, index_2, index_3, index_4])

        # Compute bilinear weights like in Treu & Koopmans (2004)
        col[~mask] = 0
        # col.at[~mask].set(0)  # Avoid accessing source_thetas out of bounds
        dist_x = (np.tile(beta_x, (4, 1)) - source_theta_x[col]) / delta_pix
        dist_y = (np.tile(beta_y, (4, 1)) - source_theta_y[col]) / delta_pix
        weight = (1 - np.abs(dist_x)) * (1 - np.abs(dist_y))

        # Make sure the weights are properly normalized
        # This step is only necessary where the mask has excluded source pixels
        norm = np.expand_dims(np.sum(weight, axis=0, where=mask), 0)
        weight = weight / norm

        if warning:
            if np.any(weight[mask] < 0):
                num_neg = np.sum((weight[mask] < 0).astype(int))
                print("Warning : {} weights are negative.".format(num_neg))

        return (row[mask], col[mask]), weight[mask]


class PlaneGrid(object):

    """
    Base class for image and source plane grids, designed for pixelated lensing operator.
    """

    def __init__(self, grid_class):
        """Initialise the grid.
        
        Parameters
        ----------
        grid_class : herculens.Coordinates.pixel_grid.PixelGrid
            PixelGrid instance
        """
        x_grid_2d, y_grid_2d = grid_class.pixel_coordinates
        self._x_grid_1d, self._y_grid_1d = util.image2array(x_grid_2d), util.image2array(y_grid_2d)
        self._delta_pix = grid_class.pixel_width
        num_pix_x, num_pix_y = grid_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._subgrid_res = 1. # grid_class.supersampling_factor

    @property
    def num_pix(self):
        return self._num_pix

    @property
    def grid_size(self):
        return self.num_pix**2

    @property
    def grid_shape(self):
        return (self._num_pix, self._num_pix)

    @property
    def delta_pix(self):
        return self._delta_pix

    @property
    def theta_x(self):
        return self._x_grid_1d

    @property
    def theta_y(self):
        return self._y_grid_1d

    @property
    def subgrid_resolution(self):
        return self._subgrid_res


class MaskedPlaneGrid(PlaneGrid):

    def __init__(self, grid_class, mask=None):
        super(MaskedPlaneGrid, self).__init__(grid_class)
        if mask is None:
            self._no_mask = True
            self._num_pix_eff = self._num_pix**2
            # self._mask_1d = np.ones(self._num_pix**2).astype(bool)
        else:
            self._no_mask = False
            self._mask_1d = mask.ravel().astype(bool)
            self._num_pix_eff = np.count_nonzero(self._mask_1d)

    @property
    def theta_x(self):
        return self._x_grid_1d if self._no_mask else self._x_grid_1d[self._mask_1d]

    @property
    def theta_y(self):
        return self._y_grid_1d if self._no_mask else self._y_grid_1d[self._mask_1d]

    @property
    def num_pix(self):
        raise self._num_pix if self._no_mask else None

    @property
    def grid_size(self):
        return self._num_pix_eff

    def place(self, values):
        if self._no_mask:
            return values
        image = jnp.zeros(self._num_pix**2)
        return image.at[self._mask_1d].set(values)

    def extract(self, values):
        if self._no_mask:
            return values
        return values[self._mask_1d]
        