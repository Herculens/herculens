# Handles coordinate grids and convolutions
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the ImSim.Numerics module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
from herculens.LensImage.Numerics.grid import RegularGrid
from herculens.LensImage.Numerics.convolution import (PixelKernelConvolution,
                                                      SubgridKernelConvolution,
                                                      GaussianConvolution)
from herculens.Util import kernel_util, util
from herculens.Util.jax_util import BilinearInterpolator, BicubicInterpolator


__all__ = ['Numerics']


class Numerics(object):
    """
    this classes manages the numerical options and computations of an image.
    The class has two main functions, re_size_convolve() and coordinates_evaluate()
    """
    def __init__(self, pixel_grid, psf, supersampling_factor=1, convolution_type='jax_scipy',
                 supersampling_convolution=False, iterative_kernel_supersampling=True,
                 supersampling_kernel_size=5, point_source_supersampling_factor=1,
                 convolution_kernel_size=None, truncation=4):
        """

        :param pixel_grid: PixelGrid() class instance
        :param psf: PSF() class instance
        :param supersampling_factor: int, factor of higher resolution sub-pixel sampling of surface brightness
        :param supersampling_convolution: bool, if True, performs (part of) the convolution on the super-sampled
        grid/pixels
        :param point_source_supersampling_factor: super-sampling resolution of the point source placing
        :param convolution_kernel_size: int, odd number, size of convolution kernel. If None, takes size of point_source_kernel
        """
        # if no super sampling, turn the supersampling convolution off
        self._psf_type = psf.psf_type
        if not isinstance(supersampling_factor, int):
            raise TypeError('supersampling_factor needs to be an integer! Current type is %s' % type(supersampling_factor))
        if supersampling_factor == 1:
            supersampling_convolution = False
        self._psf = psf

        self._pixel_width = pixel_grid.pixel_width
        nx, ny = pixel_grid.num_pixel_axes
        transform_pix2angle = pixel_grid.transform_pix2angle
        ra_at_xy_0, dec_at_xy_0 = pixel_grid.radec_at_xy_0
        self._grid = RegularGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_factor)
        self._pixel_grid = pixel_grid

        if self._psf_type == 'PIXEL':
            if supersampling_convolution is True:
                kernel_super = psf.kernel_point_source_supersampled(supersampling_factor,
                                                                    iterative_supersampling=iterative_kernel_supersampling)
                if convolution_kernel_size is not None:
                    kernel_super = self._supersampling_cut_kernel(kernel_super, convolution_kernel_size,
                                                                  supersampling_factor)
                self._conv = SubgridKernelConvolution(kernel_super, supersampling_factor,
                                                      supersampling_kernel_size=supersampling_kernel_size)
            else:
                kernel = psf.kernel_point_source
                kernel = self._supersampling_cut_kernel(kernel, convolution_kernel_size,
                                                        supersampling_factor=1)
                self._conv = PixelKernelConvolution(kernel, convolution_type=convolution_type,
                                                    output_shape=(nx, ny))

        elif self._psf_type == 'GAUSSIAN':
            pixel_scale = pixel_grid.pixel_width
            sigma = util.fwhm2sigma(psf.fwhm)
            self._conv = GaussianConvolution(sigma, pixel_scale, supersampling_factor,
                                             supersampling_convolution, truncation=truncation)
        elif self._psf_type == 'NONE':
            self._conv = None
        else:
            raise ValueError('psf_type %s not valid! Chose either NONE, GAUSSIAN or PIXEL.' % self._psf_type)
        if supersampling_convolution is True:
            self._high_res_return = True
        else:
            self._high_res_return = False

    def re_size_convolve(self, flux_array, unconvolved=False):
        """

        :param flux_array: 1d array, flux values corresponding to coordinates_evaluate
        :param array_low_res_partial: regular sampled surface brightness, 1d array
        :return: convolved image on regular pixel grid, 2d array
        """
        # add supersampled region to lower resolution on
        image_low_res, image_high_res_partial = self._grid.flux_array2image_low_high(flux_array, high_res_return=self._high_res_return)
        if unconvolved is True or self._psf_type == 'NONE':
            image_conv = image_low_res
        else:
            # convolve low res grid and high res grid
            image_conv = self._conv.re_size_convolve(image_low_res, image_high_res_partial)
        return image_conv * self._pixel_width ** 2

    def render_point_sources(self, theta_x, theta_y, amplitude):
        """Put the PSF at the locations of lensed point sources in the image plane.

        Parameters
        ----------
        theta_x : 1D array
            RA of point source positions in the image plane.
        theta_y : 1D array
            Dec of point source positions in the image plane.
        amplitude : 1D array
            Amplitudes of lensed point sources.

        Returns
        -------
        out : 2D array
            Image of pixel grid resolution where the interpolated and re-normalized
            PSF has been placed at the locations of the point sources.

        """
        result = jnp.zeros(self.original_grid.num_pixel_axes)

        # Verify inputs
        theta_x = jnp.atleast_1d(theta_x)
        theta_y = jnp.atleast_1d(theta_y)
        amplitude = jnp.atleast_1d(amplitude)

        # PSF coordinate space
        nx, ny = self._psf.kernel_point_source.shape
        width = self._psf._pixel_size
        x_coords_psf = np.linspace(-0.5 * nx * width, 0.5 * nx * width, nx)
        y_coords_psf = np.linspace(-0.5 * ny * width, 0.5 * ny * width, ny)

        x_coords_grid, y_coords_grid = self.original_grid.pixel_coordinates
        for x0, y0, amp in zip(theta_x, theta_y, amplitude):
            interp = BilinearInterpolator(x_coords_psf + x0, y_coords_psf + y0,
                                         self._psf.kernel_point_source,
                                         allow_extrapolation=False)
            ps_image = interp(x_coords_grid, y_coords_grid)
            ps_image /= ps_image.sum()
            result += amp * ps_image

        # subgrid = self._supersampling_factor
        # x_pos, y_pos = self._pixel_grid.map_coord2pix(ra_pos, dec_pos)
        # # translate coordinates to higher resolution grid
        # x_pos_subgird = x_pos * subgrid + (subgrid - 1) / 2.
        # y_pos_subgrid = y_pos * subgrid + (subgrid - 1) / 2.
        # kernel_point_source_subgrid = self._kernel_supersampled
        # # initialize grid with higher resolution
        # subgrid2d = np.zeros((self._nx*subgrid, self._ny*subgrid))
        # # add_layer2image
        # if len(x_pos) > len(amp):
        #     raise ValueError('there are %s images appearing but only %s amplitudes provided!' % (len(x_pos), len(amp)))
        # for i in range(len(x_pos)):
        #     subgrid2d = image_util.add_layer2image(subgrid2d, x_pos_subgird[i], y_pos_subgrid[i], amp[i] * kernel_point_source_subgrid)
        # # re-size grid to data resolution
        # grid2d = image_util.re_size(subgrid2d, factor=subgrid)
        # return grid2d*subgrid**2

        return result

    @property
    def grid_supersampling_factor(self):
        """

        :return: supersampling factor set for higher resolution sub-pixel sampling of surface brightness
        """
        return self._grid.supersampling_factor

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        return self._grid.coordinates_evaluate

    @staticmethod
    def _supersampling_cut_kernel(kernel_super, convolution_kernel_size, supersampling_factor):
        """

        :param kernel_super: super-sampled kernel
        :param convolution_kernel_size: size of convolution kernel in units of regular pixels (odd)
        :param supersampling_factor: super-sampling factor of convolution kernel
        :return: cut out kernel in super-sampling size
        """
        if convolution_kernel_size is not None:
            size = convolution_kernel_size * supersampling_factor
            if size % 2 == 0:
                size += 1
            kernel_cut = kernel_util.cut_psf(kernel_super, size)
            return kernel_cut
        else:
            return kernel_super

    @property
    def convolution_class(self):
        """

        :return: convolution class (can be SubgridKernelConvolution, PixelKernelConvolution or GaussianConvolution)
        """
        return self._conv

    @property
    def grid_class(self):
        """

        :return: grid class
        """
        return self._grid

    @property
    def original_grid(self):
        return self._pixel_grid
