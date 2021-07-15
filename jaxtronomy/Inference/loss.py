import numpy as np
from functools import partial
import jax.numpy as jnp
from jaxtronomy.Util.jax_util import starlet2d


class Loss(object):
    """Class that manages the loss function, defined as -[log(likelihood) + log(prior)]"""

    def __init__(self, data, image_class, param_class, 
                 likelihood_type='gaussian', 
                 regularisation_terms=['none'], lambda_regul=3,
                 prior_terms=['none']):
        self._data  = data
        self._image = image_class
        self._param = param_class

        if likelihood_type == 'gaussian':
            self._log_likelihood = self._gaussian_log_likelihood
        elif likelihood_type.lower() == 'mse':
            self._log_likelihood = self._mse_log_likelihood
        elif likelihood_type == 'chi2':
            self._log_likelihood = self._chi2_log_likelihood
        else:
            raise NotImplementedError(f"Likelihood term '{likelihood_type}' is not supported")
        
        if regularisation_terms is None or 'none' in regularisation_terms:
            self._log_regul = lambda args: 0.
        elif regularisation_terms == ['starlets_source'] \
            and self._image.SourceModel.profile_type_list == ['PIXELATED']:
            self._log_regul = self._log_regularisation_starlets_l1
            self._n_scales = int(np.log2(min(*self._data.shape)))
            npix_dirac = 2**(self._n_scales + 2)
            dirac = np.diag((np.arange(npix_dirac) == int(npix_dirac / 2)).astype(float))
            wt_dirac = starlet2d(dirac, self._n_scales)
            self._wt_norm = np.sqrt(jnp.sum(wt_dirac**2, axis=(1, 2,)))[:self._n_scales]
        else:
            raise NotImplementedError(f"Regularisation terms {regularisation_terms} is/are not supported")
        self._lambda_regul = float(lambda_regul)

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

    def _gaussian_log_likelihood(self, model):
        #noise_var = self._image.Noise.C_D_model(model)
        noise_var = self._image.Noise.C_D
        return - 0.5 * jnp.sum((self._data - model)**2 / noise_var)

    def _chi2_log_likelihood(self, model):
        #noise_var = self._image.Noise.C_D_model(model)
        noise_var = self._image.Noise.C_D
        return - jnp.mean((self._data - model)**2 / noise_var)

    def _mse_log_likelihood(self, model):
        model /= self._image.Data.pixel_width**2 # TEMP!
        return - jnp.mean((self._data - model)**2)

    def _log_regularisation_starlets_l1(self, kwargs):
        source_model = self._image.source_surface_brightness(kwargs['kwargs_source'], unconvolved=True, de_lensed=True)
        source_model /= self._image.Data.pixel_width**2 # TEMP!
        sigma_noise = self._image.Noise.background_rms  # TODO: generalise this for Poisson noise
        wt = starlet2d(source_model, self._n_scales)
        weights = jnp.expand_dims(sigma_noise * self._wt_norm, (1, 2))   # <<-- not full noise sigma !
        reg = jnp.sum(weights * jnp.abs(wt[:-1]))
        return - self._lambda_regul * reg
