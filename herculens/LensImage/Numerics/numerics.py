# Handles coordinate grids and convolutions
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the ImSim.Numerics module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
from herculens.LensImage.Numerics.grid import RegularGrid
from herculens.LensImage.Numerics.convolution import (PixelKernelConvolution, 
                                                      SubgridKernelConvolution, 
                                                      GaussianConvolution)
from herculens.Util import kernel_util, util


__all__ = ['Numerics']


class Numerics(object):
    """
    this classes manages the numerical options and computations of an image.
    The class has two main functions, re_size_convolve() and coordinates_evaluate()
    """
    def __init__(self, pixel_grid, psf, supersampling_factor=1, 
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
        self._pixel_width = pixel_grid.pixel_width
        nx, ny = pixel_grid.num_pixel_axes
        transform_pix2angle = pixel_grid.transform_pix2angle
        ra_at_xy_0, dec_at_xy_0 = pixel_grid.radec_at_xy_0
        self._grid = RegularGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_factor)
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
                self._conv = PixelKernelConvolution(kernel)

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
