# Point source model for multi-plane lensing
#
# Copyright (c) 2024, herculens developers and contributors

__author__ = 'martin-millon'

import numpy as np
import jax.numpy as jnp

try:
    from helens import LensEquationSolver
    _solver_installed = True
except ImportError:
    _solver_installed = False

__all__ = ['MPPointSourceModel']

SUPPORTED_TYPES = ['LENSED_POSITIONS', 'SOURCE_POSITION']


class MPPointSourceModel(object):
    """Collection of point sources for multi-plane lensing.

    Supports two point source types:

    - ``'LENSED_POSITIONS'``: the image-plane positions of the lensed images
      are provided directly by the user (no lens-equation solving is needed).
      The amplitude of each image is taken as-is from ``kwargs_point_source``.

    - ``'SOURCE_POSITION'``: a single position in one of the source planes is
      provided and the corresponding multiple images are found by solving the
      multi-plane lens equation.  The source plane must be specified via
      ``source_plane_index_list`` (index 1 = first plane behind the first
      deflector, index ``N_mass`` = the final source plane).  The multi-plane
      magnification at that plane is used to scale the intrinsic amplitude.
    """

    param_names = ['ra', 'dec', 'amp']

    def __init__(
        self,
        point_source_type_list,
        mp_mass_model=None,
        image_plane=None,
        source_plane_index_list=None,
    ):
        """Instantiate a multi-plane point source model.

        Parameters
        ----------
        point_source_type_list : list of str
            List of point source types. Each entry must be ``'LENSED_POSITIONS'``
            or ``'SOURCE_POSITION'``.
        mp_mass_model : MPMassModel instance, optional
            Multi-plane mass model. Required for ``'SOURCE_POSITION'`` types.
        image_plane : PixelGrid instance, optional
            Image-plane pixel grid used to build the triangular grid for the
            lens-equation solver. Required for ``'SOURCE_POSITION'`` types.
        source_plane_index_list : list of int or None, optional
            Source-plane index for each point source entry in
            ``point_source_type_list``.  For ``'SOURCE_POSITION'`` types this
            must be an integer in ``[1, N_mass]``, where ``N_mass`` is the
            number of mass planes (1 = first plane behind the first deflector,
            ``N_mass`` = the final source plane).  Entries corresponding to
            ``'LENSED_POSITIONS'`` types are ignored.  If ``None``, all point
            sources are placed on the last source plane (index ``N_mass``).
        """
        if not isinstance(point_source_type_list, list):
            raise ValueError("point_source_type_list must be a list")
        for ps_type in point_source_type_list:
            if ps_type not in SUPPORTED_TYPES:
                raise ValueError(
                    f"'{ps_type}' is not a valid point source type. "
                    f"Supported types: {SUPPORTED_TYPES}"
                )

        n = len(point_source_type_list)
        self.type_list = point_source_type_list
        self.mp_mass_model = mp_mass_model
        self.image_plane = image_plane

        if source_plane_index_list is None:
            if mp_mass_model is not None:
                default_idx = mp_mass_model.number_mass_planes
            else:
                default_idx = None
            source_plane_index_list = [default_idx] * n
        if len(source_plane_index_list) != n:
            raise ValueError(
                "source_plane_index_list must have the same length as "
                "point_source_type_list"
            )
        self.source_plane_index_list = source_plane_index_list

        for ps_type, sp_idx in zip(point_source_type_list, source_plane_index_list):
            if ps_type == 'SOURCE_POSITION':
                if not _solver_installed:
                    raise RuntimeError(
                        "A lens equation solver is required for 'SOURCE_POSITION' "
                        "point sources. Please install `helens` from "
                        "https://github.com/Herculens/helens."
                    )
                if mp_mass_model is None:
                    raise ValueError(
                        "mp_mass_model is required for 'SOURCE_POSITION' point sources."
                    )
                if image_plane is None:
                    raise ValueError(
                        "image_plane is required for 'SOURCE_POSITION' point sources."
                    )
                if sp_idx is None or sp_idx < 1:
                    raise ValueError(
                        "source_plane_index must be >= 1 for 'SOURCE_POSITION' point "
                        "sources (1 = first plane behind the first deflector)."
                    )

        # Solvers are built lazily on first use (one per SOURCE_POSITION entry)
        self._solvers = [None] * n

    # ------------------------------------------------------------------
    # Solver helpers
    # ------------------------------------------------------------------

    def _get_solver(self, i):
        """Return (lazily building) the lens equation solver for point source *i*."""
        if self._solvers[i] is None:
            sp_idx = self.source_plane_index_list[i]
            x_grid, y_grid = self.image_plane.pixel_coordinates
            self._solvers[i] = LensEquationSolver(
                x_grid, y_grid, self._make_ray_shooting_func(sp_idx)
            )
        return self._solvers[i]

    def _make_ray_shooting_func(self, source_plane_idx):
        """Return a ray-shooting callable compatible with ``LensEquationSolver``.

        The returned function has the signature
        ``(x, y, kwargs_combined) -> (beta_x, beta_y)``
        where ``kwargs_combined = [{'eta_flat': eta_flat}] + kwargs_mass``.
        The ``eta_flat`` entry allows the solver to receive the current
        multiplane distance-ratio parameters at solve time.
        """
        mp_mass_model = self.mp_mass_model

        def _ray_shoot(x, y, kwargs_combined):
            eta_flat = kwargs_combined[0]['eta_flat']
            kwargs_mass = kwargs_combined[1:]
            xs, ys = mp_mass_model.ray_shooting(
                x, y, eta_flat, kwargs_mass, N=source_plane_idx
            )
            return xs[source_plane_idx], ys[source_plane_idx]

        return _ray_shoot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_multiple_images(
        self,
        kwargs_point_source,
        eta_flat=None,
        kwargs_mass=None,
        kwargs_solver=None,
        k=None,
        with_amplitude=True,
        zero_amp_duplicates=True,
    ):
        """Compute image-plane positions and amplitudes for all point sources.

        Parameters
        ----------
        kwargs_point_source : list of dict
            One dict per point source with keys ``'ra'``, ``'dec'``, ``'amp'``.
            For ``'LENSED_POSITIONS'`` types, ``'ra'`` and ``'dec'`` are
            1-D arrays of image-plane positions and ``'amp'`` is a 1-D array
            of per-image amplitudes.  For ``'SOURCE_POSITION'`` types,
            ``'ra'`` and ``'dec'`` are scalars (source-plane position) and
            ``'amp'`` is a scalar intrinsic amplitude.
        eta_flat : jax.numpy array
            Flattened eta matrix for the multi-plane mass model.
        kwargs_mass : list of list of dict
            Per-plane mass model parameters.
        kwargs_solver : dict, optional
            Keyword arguments forwarded to the lens equation solver (e.g.
            ``nsolutions``, ``niter``, ``scale_factor``, ``nsubdivisions``).
            Required for ``'SOURCE_POSITION'`` types.
        k : int, optional
            If given, only evaluate point source number *k*. ``None`` evaluates
            all point sources.
        with_amplitude : bool, optional
            Whether to include amplitudes in the return value. Default ``True``.
        zero_amp_duplicates : bool, optional
            For ``'SOURCE_POSITION'`` types, set the amplitude of duplicate
            solver images to (near) zero. Default ``True``.

        Returns
        -------
        theta_x_list, theta_y_list : list of 1-D arrays
            Image-plane RA and Dec for each point source.
        amp_list : list of 1-D arrays
            Amplitudes of the lensed images (only returned if
            *with_amplitude* is ``True``).
        """
        theta_x_list, theta_y_list, amp_list = [], [], []

        for i in self._indices_from_k(k):
            ps_type = self.type_list[i]
            kw = kwargs_point_source[i]

            if ps_type == 'LENSED_POSITIONS':
                theta_x = jnp.atleast_1d(kw['ra'])
                theta_y = jnp.atleast_1d(kw['dec'])
                amp = jnp.atleast_1d(kw['amp'])

            else:  # SOURCE_POSITION
                sp_idx = self.source_plane_index_list[i]
                beta = jnp.array([kw['ra'], kw['dec']])
                # Pack eta_flat into the kwargs list so the solver can use it
                kwargs_combined = [{'eta_flat': eta_flat}] + list(kwargs_mass)
                kw_solver = {} if kwargs_solver is None else kwargs_solver
                solver = self._get_solver(i)
                theta, _ = solver.solve(beta, kwargs_combined, **kw_solver)
                theta_x, theta_y = theta.T
                # Multiplane magnification at the source plane
                mag = self.mp_mass_model.magnification(
                    theta_x, theta_y, eta_flat, kwargs_mass
                )[sp_idx]
                amp = kw['amp'] * jnp.abs(mag)
                if zero_amp_duplicates and kw_solver:
                    amp, theta_x, theta_y = self._zero_amp_duplicated_images(
                        amp, theta_x, theta_y, kw_solver, solver
                    )

            theta_x_list.append(theta_x)
            theta_y_list.append(theta_y)
            amp_list.append(amp)

        if with_amplitude:
            return theta_x_list, theta_y_list, amp_list
        return theta_x_list, theta_y_list

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _indices_from_k(self, k):
        inds = list(range(len(self.type_list)))
        return [k] if k in inds else inds

    @staticmethod
    def _zero_amp_duplicated_images(amp_in, theta_x_in, theta_y_in, kwargs_solver, solver):
        """Set the amplitude of duplicate images (from the solver) to near zero.

        Parameters
        ----------
        amp_in : 1-D array
            Amplitudes of all solver images.
        theta_x_in, theta_y_in : 1-D arrays
            Image-plane positions from the solver.
        kwargs_solver : dict
            Solver settings; must contain ``nsolutions``, ``niter``,
            ``scale_factor``, and ``nsubdivisions``.
        solver : LensEquationSolver instance
            The solver used to estimate positional accuracy.

        Returns
        -------
        amp_out, theta_x_out, theta_y_out : tuple of 1-D arrays
            Amplitudes (duplicates zeroed) and positions (reordered to match).
        """
        num_images = kwargs_solver['nsolutions']
        position_accuracy = solver.estimate_accuracy(
            kwargs_solver['niter'],
            kwargs_solver['scale_factor'],
            kwargs_solver['nsubdivisions'],
        )
        position_decimals = int(np.floor(-np.log10(position_accuracy))) - 1
        unique_theta_x, unique_indices = jnp.unique(
            jnp.round(theta_x_in, decimals=position_decimals),
            return_index=True,
            fill_value=False,
            size=num_images,
        )
        condition = jnp.where(unique_theta_x, True, False)
        unique_amp = amp_in[unique_indices]
        zero_amp = 1e-20  # not exactly 0 to keep autodiff well-behaved
        amp_out = jnp.where(condition, unique_amp, jnp.full(num_images, zero_amp))
        theta_x_out = theta_x_in[unique_indices]
        theta_y_out = theta_y_in[unique_indices]
        return amp_out, theta_x_out, theta_y_out
