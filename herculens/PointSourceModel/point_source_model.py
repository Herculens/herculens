# Copyright (c) 2023, herculens developers and contributors

__author__ = 'austinpeel'

from herculens.PointSourceModel.point_source import PointSource

__all__ = ['PointSourceModel']

SUPPORTED_TYPES = ['IMAGE_POSITIONS', 'SOURCE_POSITION']


class PointSourceModel(object):
    """Collection of point sources defined in the source or image plane."""

    param_names = ['ra', 'dec', 'amp']

    def __init__(self, point_source_type_list, mass_model=None, image_plane=None):
        """Instantiate a point source model.

        Parameters
        ----------
        point_source_type_list : list of str
            List of point source types to model.
        mass_model : instance of `herculens.MassModel.mass_model.MassModel`
            Model of the lensing mass used to map positions between the source
            and image planes. Default is None.
        image_plane : instance of `herculens.Coordinates.pixel_grid.PixelGrid`
            Pixel grid used for triangulation in solving the lens equation.

        """
        self.point_sources = []

        # Validate inputs
        if type(point_source_type_list) is not list:
            raise ValueError("point_source_type_list must be a list")

        # Populate point source list
        for ps_type in point_source_type_list:
            if ps_type in SUPPORTED_TYPES:
                ps = PointSource(ps_type, mass_model, image_plane)
                self.point_sources.append(ps)
            else:
                err_msg = (f"{ps_type} is not a valid point source type. " +
                           f"Supported types include {SUPPORTED_TYPES}")
                raise ValueError(err_msg)

        self.type_list = point_source_type_list

    def get_multiple_images(self, kwargs_point_source, kwargs_lens=None,
                            kwargs_solver=None, k=None, with_amplitude=True,
                            zero_amp_duplicates=True, re_compute=False):
        """Compute point source positions and amplitudes in the image plane.

        For point sources defined in the source plane, solving the lens
        equation is necessary in order to compute the corresponding (multiple)
        image plane positions.

        Parameters
        ----------
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.
        kwargs_solver : dict, optional
            Keyword arguments for the lens equation solver. Default is None.
        k : int, optional
            Index of the single point source for which to compute positions.
            If None, compute positions for all point sources.
        with_amplitude : bool, optional
            Whether to return the (magnified) amplitude of each point source.
            Default is True.
        zero_amp_duplicates : bool, optional
            If True, amplitude of duplicated images are forced to be zero.
            Note that it may affect point source ordering!.
            Default is True.
        re_compute : bool, optional
            If True, re-compute (solving the lens equation) image positions,
            even for point source models of type 'IMAGE_POSITIONS'.
            Default is False.

        Returns
        -------
        out : tuple of 2 or 3 1D arrays
            Points in the image plane given as (x-components, y-components),
            along with (optionally) their amplitudes.

        """
        theta_x, theta_y, amps = [], [], []
        for i in self._indices_from_k(k):
            ps = self.point_sources[i]
            ra, dec, amp = ps.image_positions_and_amplitudes(
                kwargs_point_source[i], kwargs_lens=kwargs_lens, kwargs_solver=kwargs_solver,
                zero_amp_duplicates=zero_amp_duplicates, re_compute=re_compute,
            )
            theta_x.append(ra)
            theta_y.append(dec)
            amps.append(amp)

        if with_amplitude:
            return theta_x, theta_y, amps

        return theta_x, theta_y

    def get_source_plane_points(self, kwargs_point_source, kwargs_lens=None,
                                k=None, with_amplitude=True):
        """Compute point source positions and amplitudes in the source plane.

        For point sources defined in the image plane, ray shooting is necessary
        to compute the corresponding source plane positions.

        Parameters
        ----------
        kwargs_point_source : list of dict
            Keyword arguments corresponding to the point source instances.
        kwargs_lens : list of dict, optional
            Keyword arguments for the lensing mass model. Default is None.
        k : int, optional
            Index of the single point source for which to compute positions.
            If None, compute positions for all point sources.
        with_amplitude : bool, optional
            Whether to return the (magnified) amplitude of each point source.
            Default is True.

        Returns
        -------
        out : tuple of 2 or 3 1D arrays
            Points in the source plane given as (x-components, y-components),
            along with (optionally) their amplitudes.

        """
        beta_x, beta_y, amps = [], [], []
        for i in self._indices_from_k(k):
            ps = self.point_sources[i]
            ra, dec = ps.source_position(kwargs_point_source[i], kwargs_lens)
            amp = ps.source_amplitude(kwargs_point_source[i], kwargs_lens)
            beta_x.append(ra)
            beta_y.append(dec)
            amps.append(amp)

        if with_amplitude:
            return beta_x, beta_y, amps

        return beta_x, beta_y
    
    def error_image_plane(self, kwargs_params, kwargs_solver):
        """This function takes as input the current parameters
        and returns distance between the predicted image positions
        based on the mean ray-traced source position, and the model image positions.
        """
        error_image = 0.
        kwargs_point_source = kwargs_params['kwargs_point_source']
        for i, ps_type in enumerate(self.type_list):
            if ps_type == 'IMAGE_POSITIONS':
                ps = self.point_sources[i]
                error_image += ps.error_image_plane(
                    kwargs_params['kwargs_point_source'][i], 
                    kwargs_params['kwargs_lens'], 
                    kwargs_solver,
                )
        return error_image
    
    def log_prob_image_plane(self, kwargs_params, kwargs_solver, **kwargs_hyperparams):
        """This function takes as input the current parameters
        and returns log-probability penalty term that enforces multiply imaged
        point sources optimized following the 'IMAGE_POSITIONS' model
        to be consistent with the lens deflection field.
        """
        log_prob = 0.
        kwargs_point_source = kwargs_params['kwargs_point_source']
        for i, ps_type in enumerate(self.type_list):
            if ps_type == 'IMAGE_POSITIONS':
                ps = self.point_sources[i]
                log_prob += ps.log_prob_image_plane(
                    kwargs_params['kwargs_point_source'][i], 
                    kwargs_params['kwargs_lens'], 
                    kwargs_solver,
                    **kwargs_hyperparams,
                )
        return log_prob
    
    def error_source_plane(self, kwargs_params):
        """This function takes as input the current parameters
        and returns distance between the predicted source positions
        based on the mean source position, and the ray-traced source positions
        from the model image positions.
        """
        error_source = 0.
        kwargs_point_source = kwargs_params['kwargs_point_source']
        for i, ps_type in enumerate(self.type_list):
            if ps_type == 'IMAGE_POSITIONS':
                ps = self.point_sources[i]
                error_source += ps.error_source_plane(
                    kwargs_params['kwargs_point_source'][i], 
                    kwargs_params['kwargs_lens'],
                )
        return error_source
    
    def log_prob_source_plane(self, kwargs_params, **kwargs_hyperparams):
        """This function takes as input the current parameters
        and returns log-probability penalty term that enforces multiply imaged
        point sources optimized following the 'IMAGE_POSITIONS' model
        to be consistent with the lens deflection field.
        """
        log_prob = 0.
        kwargs_point_source = kwargs_params['kwargs_point_source']
        for i, ps_type in enumerate(self.type_list):
            if ps_type == 'IMAGE_POSITIONS':
                ps = self.point_sources[i]
                log_prob += ps.log_prob_source_plane(
                    kwargs_params['kwargs_point_source'][i], 
                    kwargs_params['kwargs_lens'], 
                    **kwargs_hyperparams,
                )
        return log_prob

    def _indices_from_k(self, k):
        """Validate a proposed point source index.

        Parameters
        ----------
        k : int
            Proposed point source index. If k is None or is outside the range
            [0, N - 1], where N is the number of point sources, return the
            list [0, 1, ..., N - 1].

        Returns
        -------
        out : list
            Indices to take from the point source list.

        """
        inds = list(range(len(self.point_sources)))
        if k in inds:
            inds = [k]

        return inds
    