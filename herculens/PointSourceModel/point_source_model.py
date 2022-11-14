import jax.numpy as jnp
from herculens.PointSourceModel.point_source import PointSource

__all__ = ['PointSourceModel']

SUPPORTED_TYPES = ['IMAGE_POSITIONS', 'SOURCE_POSITION']

class PointSourceModel(object):
    """Collection of point sources defined in the source or image plane.

    A point source is considered to be either
    (1) a single position and amplitude defined in the source plane, or else
    (2) multiple positions and amplitudes defined in the image plane which
        correspond to a single point in the source plane.
    """
    def __init__(self, point_source_type_list, mass_model=None):
        self.point_sources = []
        if type(point_source_type_list) != list:
            raise ValueError("point_source_type_list must be a list")
        for ps_type in point_source_type_list:
            if ps_type in SUPPORTED_TYPES:
                self.point_sources.append(PointSource(ps_type, mass_model))
            else:
                raise ValueError(f"{ps_type} is not a valid point source type." +
                    f"Supported types include {SUPPORTED_TYPES}")
            # if ps_type == 'IMAGE_POSITION':
            #     self.point_source_list.append(image_position.ImagePosition(lensModel))
            # elif ps_type == 'SOURCE_POSITION':
            #     self.point_source_list.append(source_position.SourcePosition(lensModel))
            # else:
            #     raise ValueError("Valid type options are " + SUPPORTED_TYPES)

        # self.mass_model = mass_model

    def image_positions(self, kwargs_point_source, kwargs_lens=None, k=None):
        """Compute image plane positions corresponding to the point sources.

        For point sources defined in the source plane, solving the lens
        equation is necessary to compute the corresponding (multiple) image
        plane positions.

        :param kwargs_point_source: keyword arguments of the point sources
        :param kwargs_lens: keyword arguments of the mass model
        :return: arrays of image plane positions of requested point source(s)
        """
        # WARNING: k not yet implemented
        theta_x = []
        theta_y = []
        for i, ps in enumerate(self.point_sources):
            ra, dec = ps.image_positions(kwargs_point_source[i], kwargs_lens)
            theta_x += ra
            theta_y += dec
        return jnp.array(theta_x), jnp.array(theta_y)

    def source_positions(self, kwargs_point_source, kwargs_lens=None, k=None):
        """Compute source plane positions corresponding to the point sources.

        For point sources defined in the image plane, ray shooting
        is necessary to compute the corresponding source plane positions.

        :param kwargs_point_source: keyword arguments of the point sources
        :param kwargs_lens: keyword arguments of the mass model
        :return: arrays of source plane positions of requested point source(s)
        """
        # WARNING: k not yet implemented
        beta_x = []
        beta_y = []
        for i, ps in enumerate(self.point_sources):
            ra, dec = ps.source_position(kwargs_point_source[i], kwargs_lens)
            beta_x.append(ra)
            beta_y.append(dec)
        return jnp.array(beta_x), jnp.array(beta_y)

    def image_amplitudes(self, kwargs_point_source, kwargs_lens=None, k=None):
        """Compute image plane amplitudes corresponding to the point sources.

        For point sources defined in the source plane, solving the lens
        equation is necessary to compute the corresponding (multiple) image
        plane positions and magnifications.

        :param kwargs_point_source: keyword arguments of the point sources
        :param kwargs_lens: keyword arguments of the mass model
        :return: arrays of image plane amplitudes of requested point source(s)
        """
        # WARNING: k not yet implemented
        amp = []
        for i, ps in enumerate(self.point_sources):
            amp += ps.image_amplitudes(kwargs_point_source[i], kwargs_lens)
        return jnp.array(amp)

    def source_amplitudes(self, kwargs_point_source, kwargs_lens=None, k=None):
        """Compute source plane amplitudes corresponding to the point sources.

        For point sources defined in the image plane, ray shooting
        is necessary to compute the corresponding source plane positions and
        magnifications.

        :param kwargs_point_source: keyword arguments of the point sources
        :param kwargs_lens: keyword arguments of the mass model
        :return: arrays of source plane amplitudes of requested point source(s)
        """
        # WARNING: k not yet implemented
        amp = []
        for i, ps in enumerate(self.point_sources):
            amp.append(ps.source_amplitude(kwargs_point_source[i], kwargs_lens))
        return jnp.array(amp)
