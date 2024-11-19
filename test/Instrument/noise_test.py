import unittest
import numpy as np
import jax.numpy as jnp
from jax import random

from herculens.Instrument.noise import Noise

class TestNoise(unittest.TestCase):

    def setUp(self):
        self.nx, self.ny = 10, 10
        self.exposure_time = 100.0
        self.background_rms = 0.1
        self.noise_map = np.ones((self.nx, self.ny)) * 0.2
        self.variance_boost_map = np.ones((self.nx, self.ny)) * 1.5
        self.prng_key = random.PRNGKey(0)

    def test_constructor_with_noise_map(self):
        noise = Noise(self.nx, self.ny, noise_map=self.noise_map)
        self.assertTrue(np.array_equal(noise._noise_map, self.noise_map))

    def test_constructor_with_exposure_time(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time)
        self.assertEqual(noise._exp_map, self.exposure_time)

    def test_set_data(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time)
        data = np.random.rand(self.nx, self.ny)
        noise.set_data(data)
        self.assertTrue(np.array_equal(noise._data, data))

    def test_compute_noise_map_from_model(self):
        noise = Noise(self.nx, self.ny, background_rms=self.background_rms,
                       exposure_time=self.exposure_time)
        model = jnp.ones((self.nx, self.ny))
        noise.compute_noise_map_from_model(model)
        self.assertIsNotNone(noise._noise_map)
        model = jnp.ones((self.nx, self.ny))
        noise.compute_noise_map_from_model(model)  # calling it a second time to reset cached C_D matrix
        self.assertIsNotNone(noise._noise_map)

    def test_realisation(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time, background_rms=self.background_rms)
        model = jnp.ones((self.nx, self.ny))
        realisation = noise.realisation(model, self.prng_key)
        self.assertEqual(realisation.shape, (self.nx, self.ny))

    def test_variance_boost_map_property(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time)
        noise.variance_boost_map = self.variance_boost_map
        self.assertTrue(np.array_equal(noise.variance_boost_map, self.variance_boost_map))

    def test_background_rms_property(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time, background_rms=self.background_rms)
        self.assertEqual(noise.background_rms, self.background_rms)

    def test_exposure_map_property(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time)
        self.assertEqual(noise.exposure_map, self.exposure_time)

    def test_C_D_property(self):
        noise = Noise(self.nx, self.ny, noise_map=self.noise_map)
        self.assertTrue(np.array_equal(noise.C_D, self.noise_map**2))

    def test_C_D_model(self):
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time, background_rms=self.background_rms)
        model = jnp.ones((self.nx, self.ny))
        c_d_model = noise.C_D_model(model)
        self.assertEqual(c_d_model.shape, (self.nx, self.ny))
        noise.set_data(model)
        c_d_model = noise.C_D_model(model)
        self.assertEqual(c_d_model.shape, (self.nx, self.ny))

    def test_total_variance(self):
        flux = jnp.ones((self.nx, self.ny))
        variance = Noise.total_variance(flux, self.background_rms, self.exposure_time)
        self.assertEqual(variance.shape, (self.nx, self.ny))
        variance = Noise.total_variance(flux, self.background_rms, None)
        self.assertTrue(np.allclose(variance, np.full_like(flux, self.background_rms**2)))

    def test_constructor_raising_errors(self):
        with self.assertRaises(ValueError):
            noise = Noise(self.nx, self.ny, exposure_time=None, noise_map=None)

    def test_constructor_negative_exp_time(self):
        noise = Noise(self.nx, self.ny, exposure_time=-1, noise_map=None)
        noise = Noise(self.nx, self.ny, exposure_time=-1*np.ones((self.nx, self.ny)), noise_map=None)

    def test_realisation_with_missing_exposure_map(self):
        model = np.ones((10, 10))
        add_background = True
        add_object = True
        noise = Noise(self.nx, self.ny, exposure_time=None, background_rms=self.background_rms, noise_map=self.noise_map)
        with self.assertRaises(ValueError):
            noise.realisation(model, self.prng_key, add_background, add_object)

    def test_realisation_with_missing_background_rms(self):
        model = np.ones((10, 10))
        add_background = True
        add_object = True
        noise = Noise(self.nx, self.ny, background_rms=None, exposure_time=self.exposure_time, noise_map=None)
        with self.assertRaises(ValueError):
            noise.realisation(model, self.prng_key, add_background, add_object)

    def test_C_D_with_missing_data(self):
        model = np.ones((10, 10))
        noise = Noise(self.nx, self.ny, exposure_time=self.exposure_time, background_rms=self.background_rms, noise_map=None)
        with self.assertRaises(ValueError):
            noise.C_D

if __name__ == '__main__':
    unittest.main()
    