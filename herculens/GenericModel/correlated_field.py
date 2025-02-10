# This module defines a generic pixelated model that can be used in conjunction with LightModel or MassModel's pixelated profiles.
# This is a forward model based on Gaussian processes, called a correlated field.
# The implementation is a wrapper around the NIFTy.re (a.k.a JIFTy) correlated field model.

__author__ = 'aymgal'

import numpy as np
import jax
import jax.numpy as jnp
from herculens.Util import jifty_util


__all__ = ['CorrelatedField']


class CorrelatedField(object):
    """Initialize the CorrelatedField class, based on the NIFTy.re correlated field model.
    
    For more details about the `prior_` arguments, please refer to 
    the NIFTy documentation. In particular, this documentation webpage gives
    a visual intuition of the different parameters:
    https://ift.pages.mpcdf.de/nifty/user/old_nifty_getting_started_4_CorrelatedFields.html

    Parameters
    ----------
    param_suffix : str
        The suffix to be added to the field parameters name.
    mass_or_light_model : object
        The mass or light model instance.
    offset_mean : float, optional
        The global additive offset applied to the field realizations, by default np.log(1e-2).
        NOTE: if `exponentiate` is True, this value should be the log-space of the chosen offset value.
    prior_offset_std : tuple, optional
        The The mean and scatter of the log-normal of the offset, by default (0.5, 1e-6).
    prior_loglogavgslope : tuple, optional
        The mean and scatter of the log-normal distribution for the average slope
         of the power-spectrum in log-log space, by default (-4., 0.5).
    prior_fluctuations : tuple, optional
        The mean and scatter of the log-normal distribution for the fluctuations, by default (1.5, 0.8).
    prior_flexibility : object, optional
        The mean and scatter of the log-normal distribution for the flexibility, by default None.
    prior_asperity : object, optional
        The mean and scatter of the log-normal distribution for the asperity, by default None.
    cropped_border_size : int, optional
        The field can optionally be evaluated on a larger grid size,
        and then cropped to return the model in direct space, by default 0.
        This is the number of pixels added on each size of the pixelated grid.
    exponentiate : bool, optional
        Whether to return the exponential of the field, by default True. 
        For example, taking the exponential is useful to ensures non–negative values.

    Raises
    ------
    ValueError
        If the model does not have at least one Pixelated profile.
    ValueError
        If the number of pixels has not been set at creation of the LightModel or MassModel instance.
    """
    def __init__(
            self, 
            param_suffix, 
            mass_or_light_model, 
            offset_mean=np.log(1e-2),
            prior_offset_std=(0.5, 1e-6),
            prior_loglogavgslope=(-4., 0.5), 
            prior_fluctuations=(1.5, 0.8),
            prior_flexibility=None,
            prior_asperity=None,
            cropped_border_size=0,
            exponentiate=True,
        ):
        # Check that the model is pixelated
        if not mass_or_light_model.has_pixels:
            raise ValueError("The provided LightModel or MassModel must contain "
                             "at least one Pixelated profile for proper use "
                             "with the CorrelatedField model.")
        
        # retrieve the number of pixels from the LightModel instance
        # NOTE: the following is to support both when the CorrelatedField is instantiated
        # before or after the LightModel or a MassModel is passed a LensImage instance.
        # TODO: simplify this when pixelated profiles treatment is improved in LensImage.
        try:
            self._num_pix, self._num_pix_y = mass_or_light_model.pixel_grid.num_pixel_axes
        except AttributeError:
            self._num_pix = mass_or_light_model.pixel_grid_settings.get('num_pixels', None)
        else:
            if self._num_pix != self._num_pix_y:
                raise NotImplementedError("Only square pixel grids are supported for now.")
        # Pack the prior choices
        if any([p is None for p in [offset_mean, prior_offset_std, prior_loglogavgslope, prior_fluctuations]]):
            raise ValueError("Field parameters and priors `offset_mean`, `prior_offset_std`, `prior_loglogavgslope`, `prior_fluctuations` are mandatory.")
        self._kw_amplitude_offset = {
            'offset_mean': offset_mean,
            'offset_std': prior_offset_std,
        }
        self._kw_fluctuations = {
            # Amplitude of field fluctuations
            'fluctuations': prior_fluctuations,

            # Exponent of power law power spectrum component
            'loglogavgslope': prior_loglogavgslope,

            # Extra degrees of freedom
            # NOTE: I did not test much these two in a strong lensing context
            'flexibility': prior_flexibility, 
            'asperity': prior_asperity,
        }

        # Setup the correlated field model
        self._param_suffix = param_suffix
        self._cfm, self._jft_model, self._num_pix_tot, _ = jifty_util.prepare_correlated_field(
            self._param_suffix,
            self._num_pix,
            cropped_border_size,
            kwargs_amplitude=self._kw_amplitude_offset,
            kwargs_fluctuations=self._kw_fluctuations,

            # here we assume no multi-band model
            # (i.e. nothing along the wavelength dimension)
            num_pix_wl=1, border_wl=0,
            kwargs_fluctuations_wl=None,
            non_linearity='exp' if exponentiate else 'none',
        )

    def __call__(self, params):
        """Evaluate the model at the given parameters.
        The parameters keys are:
        - '{param_suffix}_field_xi': the field fluctuations
        - '{param_suffix}_field_zeromode': the zero mode of the field
        - '{param_suffix}_field_xy_dim_fluctuations': the field fluctuations (along the spatial dimensions)
        - '{param_suffix}_field_xy_dim_loglogavgslope': the log-log average slope of the power-spectrum (along the spatial dimensions)
        - '{param_suffix}_field_xy_dim_flexibility': the flexibility (along the spatial dimensions)
        - '{param_suffix}_field_xy_dim_asperity': the asperity (along the spatial dimensions)

        Parameters
        ----------
        params : Pytree
            Parameters values as a Pytree (e.g. dict).

        Returns
        -------
        jnp.Array
            Field model (in direct space), as 2d array.
        """
        return self._jft_model(params)[self._param_suffix]
    
    def model(self, params):
        """Just a handy alias"""
        return self(params)

    def numpyro_sample_pixels(self):
        """Defines the numpyro model to be used in a Pixelated (light or mass) profile.
        
        This method is only meant to be called within a numpyro model definition,
        as it defines the prior distribution (which are all standard normally distributed) for the field parameters.

        Returns
        -------
        jnp.Array
            Field model (in direct space), as 2d array.
        """
        # imports here to prevent the need for numpyro to be installed
        # if the CorrelatedField class is used in a non-numpyro context.
        import numpyro
        from numpyro.distributions import Normal, Independent
        # Base field parameters
        params = {
            f'{self._param_suffix}_field_xi': numpyro.sample(
                f'{self._param_suffix}_field_xi', 
                Independent(Normal(
                    jnp.zeros((self._num_pix_tot, self._num_pix_tot)), 
                    jnp.ones((self._num_pix_tot, self._num_pix_tot))
                ), reinterpreted_batch_ndims=2)
            ),
            f'{self._param_suffix}_field_xy_dim_fluctuations': numpyro.sample(
                f'{self._param_suffix}_field_xy_dim_fluctuations', 
                Normal(0., 1.),
            ),
            f'{self._param_suffix}_field_xy_dim_loglogavgslope': numpyro.sample(
                f'{self._param_suffix}_field_xy_dim_loglogavgslope', 
                Normal(0., 1.),
            ),
            f'{self._param_suffix}_field_zeromode': numpyro.sample(
                f'{self._param_suffix}_field_zeromode', 
                Normal(0., 1.),
            ),
        }
        # Additional optional field parameters
        if self._kw_fluctuations['flexibility'] is not None:
            params[f'{self._param_suffix}_field_xy_dim_flexibility'] = numpyro.sample(
                f'{self._param_suffix}_field_xy_dim_flexibility', 
                Normal(0., 1.),
            )
        if self._kw_fluctuations['asperity'] is not None:
            params[f'{self._param_suffix}_field_xy_dim_asperity'] = numpyro.sample(
                f'{self._param_suffix}_field_xy_dim_asperity', 
                Normal(0., 1.),
            )
        return self(params)
    
    def draw_realizations_from_prior(self, prng_key, num_samples=10):
        """Draw a random field realization from the prior distribution.

        Returns
        -------
        jnp.Array
            Field model (in direct space), as 2d array.
        """
        # imports here to prevent the need for numpyro to be installed
        # if the CorrelatedField class is used in a non-numpyro context.
        from numpyro.infer import Predictive
        def model():
            """This just to define a callable the 'numpyro way'"""
            self.numpyro_sample_pixels()
        # draw sample from the latent space (i.e. standard normal samples)
        prior_samples = Predictive(model, num_samples=num_samples)(prng_key)
        # apply the transforms to get the 
        return jax.vmap(self)(prior_samples)
    
    @property
    def correlated_field_maker(self):
        return self._cfm
    
    @property
    def num_pix_field(self):
        return self._num_pix_tot
    
    @property
    def num_pix(self):
        return self._num_pix
    