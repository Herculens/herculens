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
        psf_type : str, {'NONE', 'PIXEL', 'GAUSSIAN'}
            Type of PSF model. Default is 'NONE'.
            !! NOTE ONLY NONE AND GAUSSIAN ARE CURRENTLY IMPLEMENTED !!
        fwhm : float, optional
            Full width at half maximum, only required for 'GAUSSIAN' type.
            Default is 'NONE'.
        truncation : float, optional
            Truncation length (in units of sigma) for Gaussian model. Only
            required for 'GAUSSIAN' type. Default is None.
        pixel_size : float, optional
            Pixel width (in arcsec). Required for 'GAUSSIAN' type. Not required
            when used in combination with the ImageModel class. Default is None.
        kernel_point_source : array_like, optional
            2D array, odd length, of the centered PSF of a point source.
            Default is None.
        kernel_supersampling_factor : int, optional
            Supersampling factor of `kernel_point_source`. Default is 1.
        """
        self.psf_type = psf_type
        self._pixel_size = pixel_size
        if self.psf_type == 'GAUSSIAN':
            if fwhm is None:
                raise ValueError('`fwhm` must be set for GAUSSIAN `psf_type`!')
            self._fwhm = fwhm
            self._sigma_gaussian = util.fwhm2sigma(self._fwhm)
            self._truncation = truncation
            self._kernel_supersampling_factor = 0
        elif self.psf_type == 'PIXEL':
            if kernel_point_source is None:
                raise ValueError('kernel_point_source needs to be specified for PIXEL PSF type!')
            if len(kernel_point_source) % 2 == 0:
                raise ValueError('kernel needs to have odd axis number, not ', np.shape(kernel_point_source))
            if kernel_supersampling_factor > 1:
                self._kernel_point_source_supersampled = kernel_point_source
                self._kernel_supersampling_factor = kernel_supersampling_factor
                kernel_point_source = kernel_util.degrade_kernel(self._kernel_point_source_supersampled, self._kernel_supersampling_factor)
            self._kernel_point_source = kernel_point_source / np.sum(kernel_point_source)
        elif self.psf_type == 'NONE':
            self._kernel_point_source = np.zeros((3, 3))
            self._kernel_point_source[1, 1] = 1
        else:
            raise ValueError("psf_type %s not supported!" % self.psf_type)

    @property
    def kernel_point_source(self):
        if not hasattr(self, '_kernel_point_source'):
            if self.psf_type == 'GAUSSIAN':
                kernel_numPix = round(self._truncation * self._fwhm / self._pixel_size)
                if kernel_numPix % 2 == 0:
                    kernel_numPix += 1
                self._kernel_point_source = kernel_util.kernel_gaussian(kernel_numPix, self._pixel_size, self._fwhm)
        return self._kernel_point_source

    @property
    def kernel_pixel(self):
        """
        returns the convolution kernel for a uniform surface brightness on a pixel size

        :return: 2d numpy array
        """
        if not hasattr(self, '_kernel_pixel'):
            self._kernel_pixel = kernel_util.pixel_kernel(self.kernel_point_source, subgrid_res=1)
        return self._kernel_pixel

    def blurring_matrix(self, data_shape):
        num_pixels = data_shape[0]*data_shape[1]
        if not hasattr(self, '_blurring_matrix') or self._blurring_matrix.shape != (num_pixels, num_pixels):
            psf_kernel_2d = np.array(self.kernel_point_source)
            self._blurring_matrix = linear_util.build_convolution_matrix(psf_kernel_2d, data_shape)
        return self._blurring_matrix

    def kernel_point_source_supersampled(self, supersampling_factor, updata_cache=True, 
                                         iterative_supersampling=True):
        """
        generates (if not already available) a supersampled PSF with ood numbers of pixels centered

        :param supersampling_factor: int >=1, supersampling factor relative to pixel resolution
        :param updata_cache: boolean, if True, updates the cached supersampling PSF if generated.
         Attention, this will overwrite a previously used supersampled PSF if the resolution is changing.
        :return: super-sampled PSF as 2d numpy array
        """
        if hasattr(self, '_kernel_point_source_supersampled') and self._kernel_supersampling_factor == supersampling_factor:
            kernel_point_source_supersampled = self._kernel_point_source_supersampled
        else:
            if self.psf_type == 'GAUSSIAN':
                kernel_numPix = self._truncation / self._pixel_size * supersampling_factor
                kernel_numPix = int(round(kernel_numPix))
                if kernel_numPix % 2 == 0:
                    kernel_numPix += 1
                kernel_point_source_supersampled = kernel_util.kernel_gaussian(kernel_numPix, self._pixel_size / supersampling_factor, self._fwhm)
            elif self.psf_type == 'PIXEL':
                if iterative_supersampling is True:
                    num_iter = 5  # default value in lenstronomy
                else:
                    num_iter = 0
                kernel = kernel_util.subgrid_kernel(self.kernel_point_source, supersampling_factor, 
                                                    odd=True, num_iter=num_iter)
                n = len(self.kernel_point_source)
                n_new = n * supersampling_factor
                if n_new % 2 == 0:
                    n_new -= 1
                if hasattr(self, '_kernel_point_source_supersampled'):
                    UserWarning("Super-sampled point source kernel over-written due to different subsampling "
                                "size requested.", Warning)
                kernel_point_source_supersampled = kernel_util.cut_psf(kernel, psf_size=n_new)
            elif self.psf_type == 'NONE':
                kernel_point_source_supersampled = self._kernel_point_source
            else:
                raise ValueError('psf_type %s not valid!' % self.psf_type)
            if updata_cache is True:
                self._kernel_point_source_supersampled = kernel_point_source_supersampled
                self._kernel_supersampling_factor = supersampling_factor
        return kernel_point_source_supersampled

    def set_pixel_size(self, deltaPix):
        """
        update pixel size

        :param deltaPix: pixel size in angular units (arc seconds)
        :return: None
        """
        self._pixel_size = deltaPix
        if self.psf_type == 'GAUSSIAN' and hasattr(self, '_kernel_point_source'):
            del self._kernel_point_source
            
    @property
    def fwhm(self):
        """

        :return: full width at half maximum of kernel (in units of pixel)
        """
        return self._fwhm
        