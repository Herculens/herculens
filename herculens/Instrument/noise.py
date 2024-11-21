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
    """Class that builds a noise model, to be used in LensImage.

    Three possibilities exist, depending on what the user provides to the constructor:
    - a fixed noise map, which does not depend on any model-predicted flux.
    - an exposure time (or map) and a fixed background noise estimate,
    which will be used to estimate the noise based on an image model (see LensImage.model()).
    - an exposure time (or map) but no background noise estimate. In this case,
    the user should provide a (possibly varying) background_rms value when 
    calling the Noise.C_D_model().

    Parameters
    ----------
    nx : int
        number of data pixels along the x direction.
    ny : int
        number of data pixels along the y direction.
    exposure_time : np.array or float, optional
        exposure time, either common for all pixels or for each individual 
        data pixel. By default None
    background_rms : float, optional
        Root-mean-square value (standard deviation) of Gaussian background noise. 
        By default None.
    noise_map : np.array or int, optional
        noise standard of each individual pixel.
        If provide, overwrites any value of background_rms and exposure_time. 
        By default None.
    variance_boost_map : np.array, optional
        fixed (not model-dependent) variance boost map. By default None.
    verbose : bool, optional
        If True, outputs warning message at construction. By default True.

    Raises
    ------
    ValueError
        If neither a noise map or exposure time is provided.
    """

    def __init__(self, nx, ny, exposure_time=None, background_rms=None, 
                 noise_map=None, variance_boost_map=None, verbose=True):
        self._data = None  # TODO: is that really useful?
        self._nx, self._ny = nx, ny  # TODO: is that really useful?
        if noise_map is not None:
            assert np.shape(noise_map) == (nx, ny)
            assert np.all(noise_map > 0.)
        if exposure_time is None and noise_map is None:
            raise ValueError("Either a fixed noise map or an exposure time "
                             "(or exposure map) should be provided.")
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
        if (self._noise_map is None and self._background_rms is None
            and verbose is True):
            print("Warning: both `noise_map` and `background_rms` are None; "
                  "`background_rms` should then be given as a model parameter "
                  "to estimate the noise variance via C_D_model().")
        if variance_boost_map is None:
            variance_boost_map = np.ones((self._nx, self._ny))
        self.global_boost_map = variance_boost_map

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

    def realisation(self, model, prng_key, add_background=True, add_poisson_model=True):
            """Draws a noise realization consistent with the model,
            based on the specified parameters.

            Parameters
            ----------
            model : object
                The input model to which noise realizations will be added.
            prng_key : jax.random.PRNGKey
                The random key used for generating random numbers.
            add_background : bool, optional
                Whether to add background noise to the model. Default is True.
            add_poisson_model : bool, optional
                Whether to add Poisson noise (shot noise) to the model. Default is True.

            Returns
            -------
            float
                The total noise realization added to the model.

            Raises
            ------
            ValueError
                If `add_poisson_model` is True but `exposure_map` is None.
            ValueError
                If `add_background` is True but `background_rms` is None.
            """
            noise_real = 0.
            key1, key2 = random.split(prng_key)
            if add_poisson_model:
                if self.exposure_map is None:
                    raise ValueError("An exposure time (or map) is needed to add Poisson (shot) noise")
                noise_real += image_util.add_poisson(model, self.exposure_map, key1)
            if add_background:
                if self.background_rms is None:
                    raise ValueError("A background RMS value is needed to add background noise")
                noise_real += image_util.add_background(model, self.background_rms, key2)
            return noise_real
    
    @property
    def variance_boost_map(self):
        # NOTE: we use a setter for backward compatibility reasons
        return self.global_boost_map
    
    @variance_boost_map.setter
    def variance_boost_map(self, boost_map):
        self.global_boost_map = boost_map

    @property
    def background_rms(self):
        """
        The standard deviation ("RMS") of the background noise.
        
        NOTE: "RMS" is a misleading term; it will be changed to sigma 
        or standard deviation in the future.

        Returns
        -------
        float
            Standard deviation of the background noise

        Raises
        ------
        ValueError
            If neither a background_rms value nor a noise map is available. 
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

        Returns
        -------
        _type_
            exposure map for each pixel

        Raises
        ------
        ValueError
            If neither an exposure time (or map) nor a noise map is available.
        """
        if self._exp_map is None:
            if self._noise_map is None:
                raise ValueError("Exposure map has not been specified in Noise() class!")
        return self._exp_map

    @property
    def C_D(self):
        """Covariance matrix of all pixel values in 2d numpy array (only diagonal component)
        The covariance matrix is estimated from the data (not from any model).
        WARNING: For low count statistics, the noise in the data may lead to biased estimates of the covariance matrix.

        Returns
        -------
        jax.numpy.array
            Noise variance per pixel.

        Raises
        ------
        ValueError
            If no data image nor a noise map is available.
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
        """Returns the estimate of the variance per pixel (i.e. the diagonal
        of the data covariance matrix) with contributions from background noise
        and shot noise from the model-predicted flux.

        Parameters
        ----------
        model : jax.numpy.array
            Image model for shot noise (Poisson noise) estimation.
        background_rms : float, optional
            Standard deviation of the background noise. If not given, uses the
            (fixed) value provided to the constructor, or is ignored if a 
            (fixed) noise map has been provided.
        boost_map : jax.numpy.array, optional
            2D map (same dimensions as the model image), that contains
            multiplicative factors for the noise variance. By default 1.
        force_recompute : bool, optional
            If True, forces the use of the fixed noise map. By default False.

        Returns
        -------
        jax.numpy.array
            Noise variance per pixel corresponding to the input image model.
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
        flux : jax.numpy.array
            Pixels containing the flux from which the shot (Poisson) noise can be estimated.
        background_rms : float
            Standard deviation of the background noise.
        exposure_map : float or jax.numpy.array
            Global exposure time or exposure time per pixel.

        Returns
        -------
        jax.numpy.array
            Noise variance (background + flux-dependent shot noise) per pixel.
        """
        sigma2_bkg = background_rms**2
        if exposure_map is not None:
            flux_pos = jnp.maximum(0, flux)
            sigma2_poisson = flux_pos / exposure_map
            sigma2_tot = sigma2_bkg + sigma2_poisson
        else:
            sigma2_tot = sigma2_bkg * jnp.ones_like(flux)
        return sigma2_tot
