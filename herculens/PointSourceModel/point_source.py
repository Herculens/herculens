# Copyright (c) 2023, herculens developers and contributors

__author__ = 'austinpeel'

import numpy as np
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
        point_source_type : str
            Either 'LENSED_POSITIONS' or 'SOURCE_POSITION'.
        mass_model : instance of `herculens.MassModel.mass_model.MassModel`
            Model of the lensing mass used to map positions between the source
            and image planes. Default is None.
        image_plane : instance of `herculens.Coordinates.pixel_grid.PixelGrid`
            Pixel grid used for triangulation in solving the lens equation.

        """
        self.type = point_source_type
        self.mass_model = mass_model
        self.image_plane = image_plane

    @property
    def solver(self):
        if not hasattr(self, '_solver'):
            self._solver = LensEquationSolver(self.mass_model)
        return self._solver

    def image_positions_and_amplitudes(self, kwargs_point_source, 
                                       kwargs_lens=None, kwargs_solver=None,
                                       zero_duplicates=True):
        """Compute image plane positions and corresponding amplitudes
        of the point source, optionally "turning-off" (zeroing their amplitude)
        potentially duplicated images predicted by the lens equation solver.

        Parameters
        ----------
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.
        kwargs_solver : dict, optional
            Keyword arguments for the lens equation solver. Default is None.
        zero_duplicates : bool, optional
            If True, amplitude of duplicated images are forced to be zero.
            Note that it may affect point source ordering!.
            Default is True.
        
        Return
        ------
        theta_x, theta_y, amp : tuple of 1D arrays
            Positions (x, y) in image plane and amplitude of the lensed images.

        """
        theta_x, theta_y = self.image_positions(
            kwargs_point_source, kwargs_lens=kwargs_lens, kwargs_solver=kwargs_solver,
        )
        amp = self.image_amplitudes(
            theta_x, theta_y, kwargs_point_source, kwargs_lens=kwargs_lens,
        )
        if zero_duplicates:
            amp, theta_x, theta_y = self._zero_amp_duplicated_images(
                amp, theta_x, theta_y, kwargs_solver,
            )
        return theta_x, theta_y, amp

    def image_positions(self, kwargs_point_source, kwargs_lens=None, kwargs_solver=None):
        """Compute image plane positions of the point source.

        Parameters
        ----------
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.
        kwargs_solver : dict, optional
            Keyword arguments for the lens equation solver. Default is None.

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

            if kwargs_solver is not None:
                theta, beta = self.solver.solve(
                    self.image_plane, beta, kwargs_lens, **kwargs_solver,
                )
            else:
                theta, beta = self.solver.solve(
                    self.image_plane, beta, kwargs_lens,
                )
            return theta.T

    def image_amplitudes(self, theta_x, theta_y, kwargs_point_source, kwargs_lens=None):
        """Determine the amplitudes of the multiple images of the point source.

        Parameters
        ----------
        theta_x : array_like
            X position of points in the image plane [arcsec].
        theta_y : array_like
            Y position of points in the image plane [arcsec].
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.

        """
        amp = kwargs_point_source['amp']
        if self.type == 'IMAGE_POSITIONS':
            return jnp.atleast_1d(amp)
        elif self.type == 'SOURCE_POSITION':
            mag = self.mass_model.magnification(theta_x, theta_y, kwargs_lens)
            return amp * jnp.abs(mag)
            
    def source_position(self, kwargs_point_source, kwargs_lens=None):
        """Compute the source plane position of the point source.

        Parameters
        ----------
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.

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
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.

        """
        if self.type == 'IMAGE_POSITIONS':
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            mag = self.mass_model.magnification(theta_x, theta_y, kwargs_lens)
            amps = jnp.atleast_1d(kwargs_point_source['amp']) / abs(mag)
            return jnp.mean(amps)
        elif self.type == 'SOURCE_POSITION':
            return jnp.array(kwargs_point_source['amp'])

    def _zero_amp_duplicated_images(self, amp_in, theta_x_in, theta_y_in, kwargs_solver):
        """This function takes as input the list of multiply lensed images 
        (amplitudes and positions) and assign zero amplitude to any image 
        that have a x coordinate equal to up to `decimals` decimals.

        WARNING: this function may change the original ordering of images!

        Parameters
        ----------
        amp_in : array_like
            Amplitude of point sources
        theta_x : array_like
            X position of point sources in the image plane.
        theta_y : array_like
            Y position of point sources in the image plane.
        kwargs_solver : dict
            Keyword arguments for the LensEquation solver, used to estimate the
            accuracy of point source positions and use it to find duplicated images. 

        Returns
        -------
        amp_out, theta_x_out, theta_y_out : tuple of 3 1D arrays
            Amplitudes (potentially some being zero-ed) and positions in image plane.
        """
        # TODO: find a way not to change the image ordering (might be slower though).
        num_images = kwargs_solver['nsolutions']
        position_accuracy = self.solver.estimate_accuracy(
            self.image_plane.pixel_width,
            kwargs_solver['niter'], 
            kwargs_solver['scale_factor'], 
            kwargs_solver['nsubdivisions'], 
        )
        position_decimals = int(- np.log10(position_accuracy))
        print("position_decimals", position_decimals)
        unique_theta_x, unique_indices = jnp.unique(
            jnp.round(theta_x_in, decimals=position_decimals), 
            return_index=True, fill_value=False, size=num_images,
        )
        condition = jnp.where(unique_theta_x, True, False)
        unique_amp = amp_in[unique_indices]  # order amplitudes as the positions
        amp_out = jnp.where(condition, unique_amp, jnp.zeros(num_images))
        theta_x_out = theta_x_in[unique_indices]
        theta_y_out = theta_y_in[unique_indices]
        return amp_out, theta_x_out, theta_y_out