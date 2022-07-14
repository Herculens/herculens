import numpy as np
import jax.numpy as jnp

from herculens.RegulModel.Methods.base_method import RegulBase
from herculens.Util.jax_util import WaveletTransform


__all__ = [
    'L1StarletsBase', 
    'L1BattleLemarieWaveletBase',
    'PositivityBase',
]


class L1StarletsBase(RegulBase):

    param_names = ['lambda_hf', 'lambda_']
    lower_limit_default = {'lambda_hf': 0, 'lambda_': 0}
    upper_limit_default = {'lambda_hf': 1e8, 'lambda_': 1e8}
    fixed_default = {key: True for key in param_names}

    def __call__(self, pixels, noise_map, mask, lambda_hf=5, lambda_=3):
        """returns the log of the regularization term"""
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_levels = self._prepare_noise_levels(noise_map, mask)
        coeffs = self._wavelet.decompose(pixels)[:-1]  # ignore coarsest scale
        l1_weighted_coeffs_hf = jnp.sum(self._norms[0] * noise_levels * jnp.abs(coeffs[0]))  # first scale (i.e. high frequencies)
        l1_weighted_coeffs = jnp.sum(self._norms[1:] * noise_levels * jnp.abs(coeffs[1:]))  # other scales
        return - (lambda_hf * l1_weighted_coeffs_hf + lambda_ * l1_weighted_coeffs)

    def initialize_transform(self, model_class):
        n_pix = min(*model_class.pixelated_shape)
        n_scales = int(np.log2(n_pix))  # maximum allowed number of scales
        self._wavelet = WaveletTransform(n_scales, wavelet_type='starlet')
        wavelet_norms = self._wavelet.scale_norms[:-1]  # ignore coarsest scale
        self._norms = jnp.expand_dims(wavelet_norms, (1, 2))

    def initialize_with_lens_image(self, lens_image):
        raise NotImplementedError('Method `initialize_with_lens_image` has not been defined for this class.')

    def _prepare_noise_levels(self, noise_map, mask):
        raise NotImplementedError('Method `_prepare_noise_levels` has not been defined for this class.')


class L1BattleLemarieWaveletBase(RegulBase):

    param_names = ['lambda_']
    lower_limit_default = {'lambda_': 0}
    upper_limit_default = {'lambda_': 1e8}
    fixed_default = {key: True for key in param_names}

    def __call__(self, pixels, noise_map, mask, lambda_=10):
        """returns the log of the regularization term"""
        # TODO: generalise this for Poisson noise! but then the noise needs to be properly propagated to source plane
        noise_levels = jnp.mean(noise_map[mask == 1])
        coeffs = self._wavelet.decompose(pixels)[0]  # consider only first scale
        l1_weighted_coeffs = jnp.sum(self._norm * noise_levels * jnp.abs(coeffs))
        return - lambda_ * l1_weighted_coeffs

    def initialize_transform(self):
        n_scales = 1
        self._wavelet = WaveletTransform(n_scales, wavelet_type='battle-lemarie-3')
        self._norm = self._wavelet.scale_norms[0]  # consider only first scale

    def initialize_with_lens_image(self, lens_image):
        super().initialize_transform()

    def _prepare_noise_levels(self, noise_map, mask):
        raise NotImplementedError('Method `_prepare_noise_levels` has not been defined for this class.')


class PositivityBase(RegulBase):

    param_names = ['lambda_']
    lower_limit_default = {'lambda_': 0}
    upper_limit_default = {'lambda_': 1e8}
    fixed_default = {key: True for key in param_names}

    def __call__(self, pixels, lambda_=10):
        return - lambda_ * jnp.abs(jnp.sum(jnp.minimum(0., pixels)))
