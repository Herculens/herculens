__author__ = 'aymgal'

import jax.numpy as jnp

from herculens.RegulModel.Methods.base_pixelated \
    import L1StarletsBase, L1BattleLemarieWaveletBase


__all__ = [
    'L1StarletsSource',
    'L1BattleLemarieWaveletSource',
    'L1StarletsPotential',
    'L1BattleLemarieWaveletPotential',
]


class L1StarletsSource(L1StarletsBase):

    def initialize_with_lens_image(self, lens_image):
        super().initialize_transform(lens_image.SourceModel)

    def _prepare_noise_levels(self, noise_map, mask):
        return jnp.mean(noise_map[mask == 1])


class L1BattleLemarieWaveletSource(L1BattleLemarieWaveletBase):

    def initialize_with_lens_image(self, lens_image):
        super().initialize_transform()

    def _prepare_noise_levels(self, noise_map, mask):
        return jnp.mean(noise_map[mask == 1])


class L1StarletsPotential(L1StarletsBase):

    def __init__(self, noise_map, mask=None):
        self._noise_map = noise_map
        self._mask = mask

    def initialize_with_lens_image(self, lens_image):
        super().initialize_transform(lens_image.LensModel)

    def _prepare_noise_levels(self, noise_map, mask):
        return self._noise_map


class L1BattleLemarieWaveletPotential(L1BattleLemarieWaveletBase):

    def __init__(self, noise_map, mask=None):
        self._noise_map = noise_map
        self._mask = mask

    def _prepare_noise_levels(self, noise_map, mask):
        return self._noise_map
