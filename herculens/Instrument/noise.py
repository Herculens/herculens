# Defines the data noise model
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the Data module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import random, jit
from herculens.Util import image_util


__all__ = ['Noise']


class Noise(object):
    """
    class that deals with noise properties of imaging data
    """

    def __init__(self, nx, ny, exposure_time=None, background_rms=None, 
                 noise_map=None, variance_boost_map=None, verbose=True):
        """

        :param image_data: numpy array, pixel data values
        :param exposure_time: int or array of size the data; exposure time
        (common for all pixels or individually for each individual pixel)
        :param background_rms: root-mean-square value of Gaussian background noise
        :param noise_map: int or array of size the data; joint noise sqrt(variance) of each individual pixel.
        Overwrites meaning of background_rms and exposure_time.
        :param variance_boost_map: fixed (not model-dependent) variance boost map.
        """
        self._data = None  # TODO: is that really useful?
        self._nx, self._ny = nx, ny  # TODO: is that really useful?
        if noise_map is not None:
            assert np.shape(noise_map) == (nx, ny)
            assert np.all(noise_map > 0.)
        if exposure_time is None and noise_map is None:
            raise ValueError("Either a (fixed) noise map or an exposure time "
                             "(or map) should be provided.")
        self._noise_map = noise_map
        if exposure_time is not None:
            # make sure no negative exposure values are present no dividing by zero
            if isinstance(exposure_time, int) or isinstance(exposure_time, float):
                if exposure_time <= 1e-10:
                    exposure_time = 1e-10
            else:
                exposure_time[exposure_time <= 1e-10] = 1e-10
        self._exp_map = exposure_time
        self._background_rms = background_rms
        # if background_rms is not None:
        #     if exposure_time is not None:
        #         if background_rms * np.max(exposure_time) < 1 and verbose is True:
        #             UserWarning("sigma_b*f %s < 1 count may introduce unstable error estimates with a Gaussian "
        #                         "error function for a Poisson distribution with mean < 1." % (
        #                 background_rms * np.max(exposure_time)))
        if variance_boost_map is None:
            variance_boost_map = np.ones((self._nx, self._ny))
        self.global_boost_map = variance_boost_map  # NOTE: we use a setter for backward compatibility reasons

    def set_data(self, data):
        assert np.shape(data) == (self._nx, self._ny)
        self._reset_cache()
        self._data = data

    def compute_noise_map_from_model(self, model, as_jax_array=True):
        if self._noise_map is not None:
            UserWarning("Previous noise map will be replaced with new estimate from a model")
            self._noise_map = None
            #raise ValueError("A noise map has already been set!")
        noise_map = jnp.sqrt(self.C_D_model(model))
        self._reset_cache()
        self._noise_map = noise_map if as_jax_array else np.array(noise_map)

    def realisation(self, model, prng_key, add_gaussian=True, add_poisson=True):
        noise_real = 0.
        key1, key2 = random.split(prng_key)
        if add_poisson:
            if self.exposure_map is None:
                raise ValueError("An exposure time (or map) is needed to add Poisson noise")
            noise_real += image_util.add_poisson(model, self.exposure_map, key1)
        if add_gaussian:
            if self.background_rms is None:
                raise ValueError("An background RMS value is needed to add Poisson noise")
            noise_real += image_util.add_background(model, self.background_rms, key2)
        return noise_real
    
    @property
    def variance_boost_map(self):
        if not hasattr(self, '_boost_map'):
            return np.ones((self._nx, self._ny))
        return self._boost_map
    
    @variance_boost_map.setter
    def variance_boost_map(self, boost_map):
        self._boost_map = boost_map

    @property
    def background_rms(self):
        """

        :return: rms value of background noise
        """
        if self._background_rms is None:
            if self._noise_map is None:
                raise ValueError("rms background value as 'background_rms' not specified!")
            else:
                print("Warning: Estimating the background RMS by the median of the noise map.")
            self._background_rms = np.median(self._noise_map)
        return self._background_rms

    @property
    def exposure_map(self):
        """
        Units of data and exposure map should result in:
        number of flux counts = data * exposure_map

        :return: exposure map for each pixel
        """
        if self._exp_map is None:
            if self._noise_map is None:
                raise ValueError("Exposure map has not been specified in Noise() class!")
        return self._exp_map

    @property
    def C_D(self):
        """
        Covariance matrix of all pixel values in 2d numpy array (only diagonal component)
        The covariance matrix is estimated from the data.
        WARNING: For low count statistics, the noise in the data may lead to biased estimates of the covariance matrix.

        :return: covariance matrix of all pixel values in 2d numpy array (only diagonal component).
        """
        if not hasattr(self, '_C_D'):
            if self._noise_map is not None:
                self._C_D = self._noise_map ** 2
            else:
                if self._data is None:
                    raise ValueError("No imaging data array has been set, impossible to estimate the diagonal covariance matrix")
                self._C_D = self.total_variance(self._data, self.background_rms, self.exposure_map)
        return self._C_D

    @partial(jit, static_argnums=(0, 4))
    def C_D_model(self, model, background_rms=None, boost_map=1., 
                  force_recompute=False):
        """

        :param model: model (same as data but without noise)
        :return: estimate of the noise per pixel based on the model flux
        """
        if not force_recompute and self._noise_map is not None:
            # variance fixed by the noise map
            c_d = self._noise_map**2
        elif self._background_rms is not None:
            # variance computed based on model and fixed background RMS and exposure map
            c_d = self.total_variance(model, self._background_rms, self.exposure_map)
        else:
            # variance computed based on model, given background RMS and fixed exposure map
            c_d = self.total_variance(model, background_rms, self.exposure_map)
        return self.global_boost_map * boost_map * c_d
    
    def _reset_cache(self):
        if hasattr(self, '_C_D'):
            delattr(self, '_C_D')


    @staticmethod
    def total_variance(flux, background_rms, exposure_map):
        """Computes the diagonal of the data covariance matrix, i.e. the noise variance.

        Notes:
        - the exposure map must be positive definite. Values that deviate too much from the mean exposure time will be
            given a lower limit to not under-predict the Poisson component of the noise.
        - the flux must be positive semi-definite for the Poisson noise estimate.
            Values < 0 (Possible after mean subtraction) will not have a Poisson component in their noise estimate.

        Parameters
        ----------
        flux : _type_
            _description_
        background_rms : _type_
            _description_
        exposure_map : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if exposure_map is not None:
            flux_pos = jnp.maximum(0, flux)
            sigma2_bkg = background_rms**2
            sigma2_tot = sigma2_bkg + flux_pos / exposure_map
        else:
            sigma2_tot = background_rms**2 * jnp.ones_like(flux)
        return sigma2_tot
