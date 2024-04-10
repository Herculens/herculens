# Copyright (c) 2023, herculens developers and contributors

__author__ = 'austinpeel'

import functools
import numpy as np
import jax.numpy as jnp

try:
    from helens import LensEquationSolver
except ImportError:
    _solver_installed = False
else:
    _solver_installed = True


__all__ = ['PointSource']


class PointSource(object):
    """A point source defined in the image or source plane.

    A point source is considered to be either
    (1) a single position and amplitude defined in the source plane, or else
    (2) multiple positions and amplitudes defined in the image plane which
        correspond to a single point in the source plane.

    """

    def __init__(self, point_source_type, mass_model, image_plane):
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
        if self.type == 'SOURCE_POSITION':
            self._check_solver_install(f"type = '{self.type}'")

    @property
    def solver(self):
        if not hasattr(self, '_solver'):
            # TODO: support the argument k != None
            ray_shooting_func = functools.partial(self.mass_model.ray_shooting, k=None)
            x_grid, y_grid = self.image_plane.pixel_coordinates
            self._solver = LensEquationSolver(x_grid, y_grid, ray_shooting_func)
        return self._solver

    def image_positions_and_amplitudes(self, kwargs_point_source, 
                                       kwargs_lens=None, kwargs_solver=None,
                                       zero_amp_duplicates=True, re_compute=False):
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
        zero_amp_duplicates : bool, optional
            If True, amplitude of duplicated images are forced to be zero.
            Note that it may affect point source ordering!.
            Default is True.
        re_compute : bool, optional
            If True, re-compute (solving the lens equation) image positions,
            even for point source models of type 'IMAGE_POSITIONS'.
            Default is False.
        
        Return
        ------
        theta_x, theta_y, amp : tuple of 1D arrays
            Positions (x, y) in image plane and amplitude of the lensed images.

        """
        theta_x, theta_y = self.image_positions(
            kwargs_point_source, kwargs_lens=kwargs_lens, 
            kwargs_solver=kwargs_solver, re_compute=re_compute,
        )
        amp = self.image_amplitudes(
            theta_x, theta_y, kwargs_point_source, kwargs_lens=kwargs_lens,
        )
        if zero_amp_duplicates and self.type == 'SOURCE_POSITION':
            amp, theta_x, theta_y = self._zero_amp_duplicated_images(
                amp, theta_x, theta_y, kwargs_solver,
            )
        return theta_x, theta_y, amp

    def image_positions(self, kwargs_point_source, kwargs_lens=None, kwargs_solver=None, re_compute=False):
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
        if self.type == 'IMAGE_POSITIONS' and not re_compute:
            theta_x = jnp.atleast_1d(kwargs_point_source['ra'])
            theta_y = jnp.atleast_1d(kwargs_point_source['dec'])
            return theta_x, theta_y
        elif self.type == 'SOURCE_POSITION' or re_compute:
            if self.type == 'IMAGE_POSITIONS':  # i.e. re_compute = True
                beta_x, beta_y = self.source_position(kwargs_point_source, kwargs_lens=kwargs_lens)
            else:
                beta_x, beta_y = kwargs_point_source['ra'], kwargs_point_source['dec']
            # Solve the lens equation
            beta = jnp.array([beta_x, beta_y])
            if kwargs_solver is None:
                kwargs_solver = {}  # fall back to default lens equation solver settings
            theta, beta = self.solver.solve(
                beta, kwargs_lens, **kwargs_solver,
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
        
    def error_image_plane(self, kwargs_point_source, kwargs_lens, kwargs_solver):
        self._check_solver_install("log_prob_image_plane")
        # get the optimized image positions
        theta_x_opti = jnp.array(kwargs_point_source['ra'])
        theta_y_opti = jnp.array(kwargs_point_source['dec'])
        # find source position via ray-tracing
        beta = self.mass_model.ray_shooting(theta_x_opti, theta_y_opti, kwargs_lens)
        beta_x, beta_y = beta[0].mean(), beta[1].mean()
        beta = jnp.array([beta_x, beta_y])
        # solve lens equation to find the predicted image positions
        theta, beta = self.solver.solve(
            beta, kwargs_lens, **kwargs_solver,
        )
        theta_x_pred, theta_y_pred = theta.T
        # return departures between original and new positions
        return jnp.sqrt((theta_x_opti - theta_x_pred)**2 + (theta_y_opti - theta_y_pred)**2)
        
    def log_prob_image_plane(self, kwargs_point_source, kwargs_lens, 
                             kwargs_solver, sigma_image=1e-3):
        error_image = self.error_image_plane(kwargs_point_source, kwargs_lens, kwargs_solver)
        # penalize departures between original and new positions
        return - jnp.sum((error_image / sigma_image)**2)
    
    def error_source_plane(self, kwargs_point_source, kwargs_lens):
        # find source position via ray-tracing
        theta_x_in = jnp.array(kwargs_point_source['ra'])
        theta_y_in = jnp.array(kwargs_point_source['dec'])
        beta_x, beta_y = self.mass_model.ray_shooting(theta_x_in, theta_y_in, kwargs_lens)
        # compute distance between mean position and ray-traced positions
        return jnp.sqrt((beta_x - beta_x.mean())**2 + (beta_y - beta_y.mean())**2)
    
    def log_prob_source_plane(self, kwargs_point_source, kwargs_lens, sigma_source=1e-3):
        error_source = self.error_source_plane(kwargs_point_source, kwargs_lens)
        return - jnp.sum((error_source / sigma_source)**2)

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
            kwargs_solver['niter'], 
            kwargs_solver['scale_factor'], 
            kwargs_solver['nsubdivisions'], 
        )
        # TODO: the following choice for truncation the digits may not be general enough!
        position_decimals = np.floor(- np.log10(position_accuracy)).astype(int) - 1
        unique_theta_x, unique_indices = jnp.unique(
            jnp.round(theta_x_in, decimals=position_decimals),  # TODO: issue when original value close to zero -> rounded to exactly zero!
            return_index=True, 
            fill_value=False,   # effectively zero
            size=num_images,
        )
        condition = jnp.where(unique_theta_x, True, False)
        unique_amp = amp_in[unique_indices]  # order amplitudes as the positions
        zero_amp = 1e-20  # not exactly 0 to avoid problems with autodiff gradients
        amp_out = jnp.where(condition, unique_amp, jnp.full(num_images, zero_amp))
        theta_x_out = theta_x_in[unique_indices]
        theta_y_out = theta_y_in[unique_indices]
        return amp_out, theta_x_out, theta_y_out

    def _check_solver_install(self, feature):
        if not _solver_installed:
            raise RuntimeError(f"A lens equation solver is required for the "
                               f"require point source modeling feature ('{feature}'). "
                               f"Please install `helens` from https://github.com/Herculens/helens.")
        