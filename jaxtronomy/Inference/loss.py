import numpy as np
from functools import partial
import jax.numpy as jnp
from jaxtronomy.Util.jax_util import WaveletTransform


class Loss(object):
    """
    Class that manages the loss function, defined as:
    L = - log(likelihood) - log(prior) - log(regularisation)
    """

    def __init__(self, data, image_class, param_class, 
                 likelihood_type='chi2', prior_terms=['none'],
                 regul_terms=['none'], regul_strengths=[3]):
        self._data  = data
        self._image = image_class
        self._param = param_class

        if likelihood_type == 'chi2':
            self._log_likelihood = self._log_likelihood_chi2
        elif likelihood_type == 'l2_norm':
            self._log_likelihood = self._log_likelihood_l2
        else:
            raise NotImplementedError(f"Likelihood term '{likelihood_type}' is not supported")
        
        if regul_terms is None or 'none' in regul_terms:
            self._log_regul = lambda args: 0.
        elif regul_terms == ['starlets_source'] \
            and self._image.SourceModel.profile_type_list == ['PIXELATED']:
            self._log_regul = self._log_regularisation_starlets_l1
            n_scales = int(np.log2(min(*self._data.shape)))
            self._starlet = WaveletTransform(n_scales, wavelet_type='starlet')
            sigma_noise = self._image.Noise.background_rms  # TODO: generalise this for Poisson noise
            self._st_weights = jnp.expand_dims(sigma_noise * self._starlet.scale_norms, (1, 2))   # <<-- not full noise sigma !
            self._st_lambda = float(regul_strengths[0])
        else:
            raise NotImplementedError(f"Regularisation terms {regul_terms} is/are not supported")
        
        if prior_terms is None or 'none' in prior_terms:
            self._log_prior = lambda args: 0.
        elif prior_terms == ['uniform']:
            self._log_prior = self._param.log_prior_uniform
        elif prior_terms == ['gaussian']:
            self._log_prior = self._param.log_prior_gaussian
        elif 'gaussian' in prior_terms and 'uniform' in prior_terms:
            self._log_prior = self._param.log_prior
        else:
            raise NotImplementedError(f"Prior terms {prior_terms} is/are not supported")

    def __call__(self, args):
        return self.loss(args)

    def loss(self, args):
        """defined as the negative log(likelihood*prior*regularisation)"""
        kwargs = self._param.args2kwargs(args)
        log_L = self.log_likelihood(self._image.model(**kwargs))
        log_R = self.log_regularisation(kwargs)
        log_P = self.log_prior(args)
        return - log_L - log_R - log_P

    def log_likelihood(self, model):
        return self._log_likelihood(model)

    def log_regularisation(self, kwargs):
        return self._log_regul(kwargs)

    def log_prior(self, args):
        return self._log_prior(args)

    def _log_likelihood_chi2(self, model):
        #noise_var = self._image.Noise.C_D_model(model)
        noise_var = self._image.Noise.C_D
        return - jnp.mean((self._data - model)**2 / noise_var)

    def _log_likelihood_l2(self, model):
        model /= self._image.Data.pixel_width**2 # TEMP!
        return - 0.5 * jnp.sum((self._data - model)**2)

    def _log_regularisation_starlets_l1(self, kwargs):
        # TODO: fix issue with JAX and .source_surface_brightness() method
        #source_model = self._image.source_surface_brightness(kwargs['kwargs_source'], unconvolved=True, de_lensed=True)
        #source_model /= self._image.Data.pixel_width**2 # TEMP!
        source_model = kwargs['kwargs_source'][0]['image']
        st = self._starlet.decompose(source_model)
        st_weighted_l1 = jnp.sum(self._st_weights * jnp.abs(st[:-1]))
        return - self._st_lambda * st_weighted_l1
