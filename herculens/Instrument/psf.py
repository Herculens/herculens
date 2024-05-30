# Defines the point spread function model
#
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Data module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import warnings
import numpy as np
from herculens.Util import util, kernel_util
from utax.convolution.functions import build_convolution_matrix


__all__ = ['PSF']


class PSF(object):
    """Point Spread Function class.

    This class describes and manages products used to perform the PSF modeling,
    including convolution for extended surface brightness and painting on the
    PSF for point sources.

    """

    def __init__(self, psf_type='NONE', fwhm=None, truncation=5,
                 pixel_size=None, kernel_point_source=None,
                 kernel_supersampling_factor=1,
                 variance_boost_map=None):
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
            Pixel width (in arcsec). Required for 'GAUSSIAN' type. Default is None.
        kernel_point_source : array_like, optional
            2D array of odd length representing the centered PSF. Required
            for 'PIXEL' type. Default is None.
        kernel_supersampling_factor : int, optional
            For a 'PIXEL' type PSF, this parameter specifies the factor by
            which the provided `kernel_point_source` has been supersampled.
            Default is 1.
        variance_boost_map : array_like or float, optional
            Single floating point value or 2D array of floating point values 
            that multiply the model noise variance associated to a given pixel value.
            For the 'GAUSSIAN' model, only the single value option is currently supported.
        """
        self.psf_type = psf_type
        self._pixel_size = pixel_size
        if variance_boost_map is None:
            variance_boost_map = 1.

        if self.psf_type == 'GAUSSIAN':
            # Validate required inputs
            if fwhm is None:
                raise ValueError("Must set `fwhm` if `psf_type='GAUSSIAN'`")
            if pixel_size is None:
                raise ValueError("Must set `pixel_size` if `psf_type='GAUSSIAN'`")

            self._fwhm = fwhm
            self._sigma_gaussian = util.fwhm2sigma(self._fwhm)
            self._truncation = truncation
            self._kernel_supersampling_factor = 0
            kernel = self.compute_gaussian_kernel(self._pixel_size, self.fwhm,
                                                  self._truncation)
            self._kernel_point_source = kernel
            self._var_boost_map = variance_boost_map

        elif self.psf_type == 'PIXEL':
            # Validate required inputs
            if kernel_point_source is None:
                raise ValueError(
                    'Must set `kernel_point_source` for PIXEL `psf_type`')
            if len(kernel_point_source) % 2 == 0:
                raise ValueError(
                    'kernel needs to have odd axis number, not ', np.shape(kernel_point_source))
            self._kernel_supersampling_factor = kernel_supersampling_factor
            if kernel_supersampling_factor > 1:
                self._kernel_point_source_supersampled = kernel_point_source
                kernel_point_source = kernel_util.degrade_kernel(
                    self._kernel_point_source_supersampled, self._kernel_supersampling_factor)
            self._kernel_point_source = kernel_point_source / \
                np.sum(kernel_point_source)
            if isinstance(variance_boost_map, (float, int)):
                variance_boost_map = np.full_like(
                    self._kernel_point_source.shape,
                    variance_boost_map,
                )
            elif variance_boost_map.shape != self._kernel_point_source.shape:
                raise ValueError("Variance boost map should have the same shape as the PSF kernel.")
            self._var_boost_map = variance_boost_map
            
        elif self.psf_type == 'NONE':
            self._kernel_point_source = np.zeros((3, 3))
            self._kernel_point_source[1, 1] = 1
            self._var_boost_map = variance_boost_map
    
        else:
            raise ValueError("psf_type %s not supported" % self.psf_type)

    @property
    def kernel_point_source(self):
        if not hasattr(self, '_kernel_point_source'):
            return None
        return self._kernel_point_source

    @property
    def kernel_supersampling_factor(self):
        return self._kernel_supersampling_factor

    @property
    def variance_boost_map(self):
        if not hasattr(self, '_var_boost_map'):
            return None
        return self._var_boost_map

    def kernel_point_source_supersampled(self, supersampling_factor,
                                         iterative_supersampling=False,
                                         update_cache=False):
        """Generate a supersampled PSF.

        The resulting 2D array has the supersampled PSF at the center of a grid
        with odd dimension.

        Parameters
        ----------
        supersampling_factor : int
            Supersampling factor relative to the pixel size.
        iterative_supersampling : bool
            If True, use 5 iterations when supersampling.
        update_cache : bool
            If True, update (and overwrite, if present) the cached supersampled PSF.

        """
        if (hasattr(self, '_kernel_point_source_supersampled') and
                self._kernel_supersampling_factor == supersampling_factor and 
                not update_cache):
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
    
    def blurring_matrix(self, data_shape):
        num_pixels = data_shape[0] * data_shape[1]
        if not hasattr(self, '_blurring_matrix') or self._blurring_matrix.shape != (num_pixels, num_pixels):
            psf_kernel_2d = np.array(self.kernel_point_source)
            self._blurring_matrix = build_convolution_matrix(
                psf_kernel_2d, data_shape)
        return self._blurring_matrix

    def set_pixel_size(self, pixel_size):
        """Update pixel size.

        Parameters
        ----------
        pixel_size : float
            New pixel size in angular units (arc seconds).

        Notes
        -----
        The `kernel_point_source` attribute is (re)computed according to the
        new pixel size if the PSF type is GAUSSIAN.

        """
        self._pixel_size = pixel_size

        if self.psf_type == 'GAUSSIAN':
            kernel = self.compute_gaussian_kernel(self._pixel_size, self.fwhm,
                                                  self._truncation)
            self._kernel_point_source = kernel

    def compute_gaussian_kernel(self, pixel_size, fwhm, truncation):
        """Compute a Gaussian kernel matrix to serve as PSF.

        Parameters
        ----------
        pixel_size : float
            Pixel size in angular units (arc seconds).
        fwhm : float
            Full width at half maximum of the Gaussian in pixel units.
        truncation : float
            Truncation length (in units of sigma) for the Gaussian.

        Returns
        -------
        out : array
            2D Gaussian kernel matrix.

        """
        # Determine the number of pixels per side
        npix = round(self._truncation * self.fwhm / self._pixel_size)
        # Ensure an odd number
        npix += 1 - npix % 2
        # Evaluate the 2D Gaussian at the pixel positions
        return kernel_util.kernel_gaussian(npix, self._pixel_size, self.fwhm)

    @property
    def fwhm(self):
        """Full width at half maximum of a Gaussian kernel in pixel units."""
        return self._fwhm
