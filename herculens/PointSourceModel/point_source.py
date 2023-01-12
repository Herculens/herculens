__author__ = 'austinpeel'

import jax.numpy as jnp
from herculens.MassModel.lens_equation import LensEquationSolver

__all__ = ['PointSource']

class PointSource(object):
    """A point source defined in the image or source plane.

    A point source is considered to be either
    (1) a single position and amplitude defined in the source plane, or else
    (2) multiple positions and amplitudes defined in the image plane which
        correspond to a single point in the source plane.

    """
    def __init__(self, point_source_type, mass_model=None, image_plane=None):
        """Instantiate a point source.

        Parameters
        ----------
        TODO

        """
        self.type = point_source_type
        self.mass_model = mass_model
        self.image_plane = image_plane

    def image_positions(self, kwargs_point_source, kwargs_lens=None):
        """Compute image plane positions of the point source.

        Parameters
        ----------
        TODO

        """
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            return theta_x, theta_y
        elif self.type == 'SOURCE_POSITION':
            # Solve the lens equation
            beta_x = kwargs_point_source['ra']
            beta_y = kwargs_point_source['dec']
            beta = jnp.array([beta_x, beta_y])

            if not hasattr(self, '_solver'):
                self._solver = LensEquationSolver(self.mass_model)

            theta, beta = self._solver.solve(self.image_plane, beta, kwargs_lens)
            return theta.T

    def image_amplitudes(self, theta_x, theta_y, kwargs_point_source, kwargs_lens=None):
        """Determine the amplitudes of the multiple images of the point source.

        Parameters
        ----------
        TODO

        """
        amp = kwargs_point_source['amp']
        if self.type is 'IMAGE_POSITIONS':
            return jnp.atleast_1d(amp)
        elif self.type is 'SOURCE_POSITION':
            mag = self.mass_model.magnification(theta_x, theta_y, kwargs_lens)
            return amp * jnp.abs(mag)

    def source_position(self, kwargs_point_source, kwargs_lens=None):
        """Compute the source plane position of the point source.

        Parameters
        ----------
        TODO

        """
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.array(kwargs_point_source['ra'])
            theta_y = jnp.array(kwargs_point_source['dec'])
            beta = self.mass_model.ray_shooting(theta_x, theta_y, kwargs_lens)
            return jnp.mean(beta[0]), jnp.mean(beta[1])
        elif self.type == 'SOURCE_POSITION':
            beta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            beta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            return beta_x, beta_y

    def source_amplitude(self, kwargs_point_source, kwargs_lens=None):
        """Determine the amplitude of the point source in the source plane.

        Parameters
        ----------
        TODO

        """
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            mag = self.mass_model.magnification(theta_x, theta_y, kwargs_lens)
            amps = jnp.atleast_1d(kwargs_point_source['amp']) / abs(mag)
            return jnp.mean(amps)
        elif self.type == 'SOURCE_POSITION':
            return jnp.array(kwargs_point_source['amp'])
