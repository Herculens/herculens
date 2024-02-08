# Handles different convolution methods
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the ImSim.Numerics module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from utax.convolution.classes import GaussianFilter
from utax.convolution.functions import build_convolution_matrix

from herculens.Util import util, kernel_util, image_util


__all__ = ['PixelKernelConvolution', 'SubgridKernelConvolution', 'GaussianConvolution']


class PixelKernelConvolution(object):
    """
    class to compute convolutions for a given pixelized kernel
    """

    _conv_types = ['jax_scipy_fft', 'jax_scipy', 'matrix']

    def __init__(self, kernel, convolution_type='jax_scipy_fft', output_shape=None):
        """

        :param kernel: 2d array, convolution kernel
        """
        self._kernel = kernel
        if convolution_type not in self._conv_types:
            raise ValueError(f"Convolution type '{convolution_type}' not supported "
                             f"(should in {self._conv_types}).")
        self._conv_type = convolution_type
        if self._conv_type == 'matrix':
            if output_shape is None:
                raise ValueError("An output shape must be provided to build the convolution matrix.")
            self._conv_matrix  = build_convolution_matrix(kernel, output_shape)
            self._output_shape = output_shape
        else:
            self._conv_matrix  = None
            self._output_shape = None

    def pixel_kernel(self, num_pix=None):
        """
        access pixelated kernel

        :param num_pix: size of returned kernel (odd number per axis). If None, return the original kernel.
        :return: pixel kernel centered
        """
        if num_pix is not None:
            return kernel_util.cut_psf(self._kernel, num_pix)
        return self._kernel

    def convolution2d(self, image):
        """

        :param image: 2d array (image) to be convolved
        :return: fft convolution
        """
        if self._conv_type == 'jax_scipy_fft':
            return jsp.signal.fftconvolve(image, self._kernel, mode='same')
        elif self._conv_type == 'jax_scipy':
            return jsp.signal.convolve2d(image, self._kernel, mode='same')
        elif self._conv_type == 'matrix':
            return self._conv_matrix.dot(image.flatten()).reshape(*self._output_shape)

    def re_size_convolve(self, image_low_res, image_high_res=None):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        return self.convolution2d(image_low_res)

    @property
    def convolution_matrix(self):
        return self._conv_matrix



class SubgridKernelConvolution(object):
    """
    class to compute the convolution on a supersampled grid with partial convolution computed on the regular grid
    """
    def __init__(self, kernel_supersampled, supersampling_factor, supersampling_kernel_size=None):
        """

        :param kernel_supersampled: kernel in supersampled pixels
        :param supersampling_factor: supersampling factor relative to the image pixel grid
        :param supersampling_kernel_size: number of pixels (in units of the image pixels) that are convolved with the
        supersampled kernel
        """
        n_high = len(kernel_supersampled)
        self._supersampling_factor = supersampling_factor
        numPix = int(n_high / self._supersampling_factor)
        if supersampling_kernel_size is None:
            kernel_low_res, kernel_high_res = np.zeros((3, 3)), kernel_supersampled
            self._low_res_convolution = False
        else:
            kernel_low_res, kernel_high_res = kernel_util.split_kernel(kernel_supersampled, supersampling_kernel_size,
                                                                       self._supersampling_factor)
            self._low_res_convolution = True
        self._low_res_conv = PixelKernelConvolution(kernel_low_res)
        self._high_res_conv = PixelKernelConvolution(kernel_high_res)

    def convolution2d(self, image):
        """

        :param image: 2d array (high resoluton image) to be convolved and re-sized
        :return: convolved image
        """

        image_high_res_conv = self._high_res_conv.convolution2d(image)
        image_resized_conv = image_util.re_size(image_high_res_conv, self._supersampling_factor)
        if self._low_res_convolution is True:
            image_resized = image_util.re_size(image, self._supersampling_factor)
            image_resized_conv += self._low_res_conv.convolution2d(image_resized)
        return image_resized_conv

    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_high_res_conv = self._high_res_conv.convolution2d(image_high_res)
        image_resized_conv = image_util.re_size(image_high_res_conv, self._supersampling_factor)
        if self._low_res_convolution is True:
            image_resized_conv += self._low_res_conv.convolution2d(image_low_res)
        return image_resized_conv


class GaussianConvolution(object):
    """
    class to perform a convolution a 2d Gaussian
    """

    def __init__(self, sigma, pixel_scale, supersampling_factor=1,
                 supersampling_convolution=False, truncation=2):
        self._sigma = sigma / pixel_scale
        if supersampling_convolution is True:
            self._sigma *= supersampling_factor
        self._truncation = truncation
        self._pixel_scale = pixel_scale
        self._supersampling_factor = supersampling_factor
        self._supersampling_convolution = supersampling_convolution
        self._gaussian_filter = GaussianFilter(self._sigma, self._truncation)

    def convolution2d(self, image):
        """
        2d convolution

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """
        return self._gaussian_filter(image)

    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        if self._supersampling_convolution:
            image_high_res_conv = self.convolution2d(image_high_res)
            image_resized_conv = image_util.re_size(image_high_res_conv, self._supersampling_factor)
        else:
            image_resized_conv = self.convolution2d(image_low_res)
        return image_resized_conv

    def pixel_kernel(self, num_pix):
        """
        computes a pixelized kernel from the MGE parameters

        :param num_pix: int, size of kernel (odd number per axis)
        :return: pixel kernel centered
        """
        x, y = util.make_grid(numPix=num_pix, deltapix=self._pixel_scale)
        sigma = self._sigma
        diff_square = (x - center_x) ** 2 / sigma**2 + (y - center_y) ** 2 / sigma**2
        kernel = jnp.exp(- diff_square / 2.)
        kernel = util.array2image(kernel)
        return kernel / jnp.sum(kernel)
