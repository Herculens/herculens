import jax.numpy as jnp

__all__ = ['PointSource']

class PointSource(object):
    def __init__(self, point_source_type, mass_model=None):
        self.type = point_source_type
        self.mass_model = mass_model

    def image_positions(self, kwargs_point_source, kwargs_lens=None):
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            return list(theta_x), list(theta_y)
        elif self.type == 'SOURCE_POSITION':
            # Solve the lens equation
            print("Can't yet solve the lens equation!")
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            return list(theta_x), list(theta_y)

    def source_position(self, kwargs_point_source, kwargs_lens=None):
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.array(kwargs_point_source['ra'])
            theta_y = jnp.array(kwargs_point_source['dec'])
            beta_x, beta_y = self.mass_model.ray_shooting(theta_x, theta_y, kwargs_lens)
            return jnp.mean(beta_x), jnp.mean(beta_y)
        elif self.type == 'SOURCE_POSITION':
            return kwargs_point_source['ra'], kwargs_point_source['dec']

    def image_amplitudes(self, kwargs_point_source, kwargs_lens=None):
        if self.type == 'IMAGE_POSITIONS':
            return list(jnp.atleast_1d(kwargs_point_source['amp']))
        elif self.type == 'SOURCE_POSITION':
            # Solve the lens equation
            print("Can't yet solve the lens equation!")
            amp = jnp.atleast_1d(kwargs_point_source['amp'])
            return list(amp)

    def source_amplitude(self, kwargs_point_source, kwargs_lens=None):
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            mag = self.mass_model.magnification(theta_x, theta_y, kwargs_lens)
            amps = jnp.atleast_1d(kwargs_point_source['amp']) / jnp.abs(mag)
            return jnp.mean(amps)
        elif self.type == 'SOURCE_POSITION':
            return kwargs_point_source['amp']
