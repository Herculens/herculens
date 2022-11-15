# Defines the point spread function model
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Data module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import warnings
import numpy as np
from herculens.Util import util, kernel_util, linear_util


__all__ = ['PSF']


class PSF(object):
    """Point Spread Function class.

    This class describes and manages products used to perform the PSF modeling,
    including convolution for extended surface brightness and painting on the
    PSF for point sources.

    """
    def __init__(self, psf_type='NONE', fwhm=None, truncation=5,
                 pixel_size=None, kernel_point_source=None,
                 kernel_supersampling_factor=1):
        """Create a PSF object.

        Parameters
        ----------
        psf_type : str, one of {'NONE', 'PIXEL', 'GAUSSIAN'}
            Type of PSF model. Default is 'NONE'.
        fwhm : float, optional
            Full width at half maximum, only required for 'GAUSSIAN' type.
            Default is 'NONE'.
        truncation : float, optional
            Truncation length (in units of sigma) for Gaussian model. Only
            required for 'GAUSSIAN' type. Default is None.
        pixel_size : float, optional
            Pixel width (in arcsec). Required for 'GAUSSIAN' type. Not required
            when used in combination with the LensImage class. Default is None.
        kernel_point_source : array_like, optional
            2D array of odd length representing the centered PSF. Required
            for 'PIXEL' type. Default is None.
        kernel_supersampling_factor : int, optional
            For a 'PIXEL' type PSF, this parameter specifies the factor by
            which the provided `kernel_point_source` has been supersampled.
            Default is 1.

        """
        self.psf_type = psf_type
        self._pixel_size = pixel_size

        if self.psf_type == 'GAUSSIAN':
            # Validate required inputs
            if fwhm is None:
                raise ValueError('Must set `fwhm` for GAUSSIAN `psf_type`')

            self._fwhm = fwhm
            self._sigma_gaussian = util.fwhm2sigma(self._fwhm)
            self._truncation = truncation
            self._kernel_supersampling_factor = 0
        elif self.psf_type == 'PIXEL':
            # Validate required inputs
            if kernel_point_source is None:
                raise ValueError('Must set `kernel_point_source` for PIXEL `psf_type`')
            if len(kernel_point_source) % 2 == 0:
                raise ValueError('Kernel must have odd axis number, not ',
                    np.shape(kernel_point_source))

            self._kernel_point_source = kernel_point_source
            self._kernel_supersampling_factor = kernel_supersampling_factor
            if kernel_supersampling_factor > 1:
                self._kernel_point_source_supersampled = kernel_point_source
                subsampled = kernel_util.degrade_kernel(kernel_point_source,
                    kernel_supersampling_factor)
                self._kernel_point_source = subsampled / np.sum(subsampled)
        elif self.psf_type == 'NONE':
            self._kernel_point_source = np.zeros((3, 3))
            self._kernel_point_source[1, 1] = 1
        else:
            raise ValueError("psf_type %s not supported" % self.psf_type)

    @property
    def kernel_point_source(self):
        if hasattr(self, '_kernel_point_source'):
            return self._kernel_point_source

        if self._pixel_size is None:
            raise ValueError('Must first set `pixel_size`')

        if self.psf_type == 'GAUSSIAN':
            npix = round(self._truncation * self._fwhm / self._pixel_size)
            # Ensure an odd number of pixels
            npix += 1 - npix % 2
            kernel = kernel_util.kernel_gaussian(npix, self._pixel_size,
                                                 self._fwhm)
            self._kernel_point_source = kernel

        return self._kernel_point_source

    @property
    def kernel_pixel(self):
        """
        returns the convolution kernel for a uniform surface brightness on a pixel size

        :return: 2d numpy array
        """
        # WARNING kernel_util.pixel_kernel() is not implemented
        # Check where this method is used elsewhere to determine if it can be removed
        if not hasattr(self, '_kernel_pixel'):
            self._kernel_pixel = kernel_util.pixel_kernel(self.kernel_point_source, subgrid_res=1)
        return self._kernel_pixel

    def blurring_matrix(self, data_shape):
        num_pixels = data_shape[0] * data_shape[1]
        if not hasattr(self, '_blurring_matrix') or self._blurring_matrix.shape != (num_pixels, num_pixels):
            psf_kernel_2d = np.array(self.kernel_point_source)
            self._blurring_matrix = linear_util.build_convolution_matrix(psf_kernel_2d, data_shape)
        return self._blurring_matrix

    def kernel_point_source_supersampled(self, supersampling_factor, update_cache=True,
                                         iterative_supersampling=True):
        """Generate a supersampled PSF.

        The resulting 2D array has the supersampled PSF at the center of a grid
        with odd dimension.

        Parameters
        ----------
        supersampling_factor : int
            Supersampling factor relative to the pixel size.
        update_cache : bool
            If True, update (and overwrite, if present) the cached supersampled PSF.
        iterative_supersampling : bool
            If True, use 5 iterations when supersampling.

        """
        if (hasattr(self, '_kernel_point_source_supersampled') and
            self._kernel_supersampling_factor == supersampling_factor):
            return self._kernel_point_source_supersampled

        if self.psf_type == 'GAUSSIAN':
            npix = self._truncation / self._pixel_size * supersampling_factor
            npix = int(round(npix))
            # Ensure an odd number of pixels
            npix += 1 - npix % 2
            pixel_size = self._pixel_size / supersampling_factor
            result = kernel_util.kernel_gaussian(npix, pixel_size, self._fwhm)
        elif self.psf_type == 'PIXEL':
            num_iter = 5 if iterative_supersampling else 0
            kernel = kernel_util.subgrid_kernel(self.kernel_point_source,
                supersampling_factor, odd=True, num_iter=num_iter)
            npix = len(self.kernel_point_source) * supersampling_factor
            npix -= (1 - npix % 2)
            if hasattr(self, '_kernel_point_source_supersampled'):
                warnings.warn("Overwriting supersampled point source kernel " +
                            "due to different subsampling size.")
            result = kernel_util.cut_psf(kernel, psf_size=npix)
        elif self.psf_type == 'NONE':
            result = self._kernel_point_source
        else:
            raise ValueError(f'psf_type {self.psf_type} not valid.')

        if update_cache:
            self._kernel_point_source_supersampled = result
            self._kernel_supersampling_factor = supersampling_factor

        return result

    def set_pixel_size(self, pixel_size):
        """Update pixel size.

        Parameters
        ----------
        pixel_size : float
            New pixel size in angular units (arc seconds).

        """
        self._pixel_size = pixel_size
        if self.psf_type == 'GAUSSIAN' and hasattr(self, '_kernel_point_source'):
            del self._kernel_point_source

    @property
    def fwhm(self):
        """Full width at half maximum of a Gaussian kernel in pixel units."""
        return self._fwhm
