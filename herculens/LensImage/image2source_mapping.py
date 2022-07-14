import numpy as np

__all__ = ['Image2SourceMapping']


class Image2SourceMapping(object):
    """
    this class handles multiple source planes and performs the computation of predicted surface brightness at given
    image positions.
    The class is enable to deal with an arbitrary number of different source planes. There are two different settings:

    Single lens plane modelling:
    In case of a single deflector, herculens models the reduced deflection angles
    (matched to the source plane in single source plane mode). Each source light model can be added a number
    (scale_factor) that rescales the reduced deflection angle to the specific source plane.

    Multiple lens plane modelling:
    The multi-plane lens modelling requires the assumption of a cosmology and the redshifts of the multiple lens and
    source planes. The backwards ray-tracing is performed and stopped at the different source plane redshift to compute
    the mapping between source to image plane.
    """

    def __init__(self, lens_model, source_model):
        """

        :param lens_model: herculens LensModel() class instance
        :param source_model: LightModel () class instance
        The lightModel includes:
        - source_scale_factor_list: list of floats corresponding to the rescaled deflection angles to the specific source
         components. None indicates that the list will be set to 1, meaning a single source plane model (in single lens plane mode).
        - source_redshift_list: list of redshifts of the light components (in multi lens plane mode)
        """
        self._lightModel = source_model
        self._lensModel = lens_model

    def image2source(self, x, y, kwargs_lens, index_source):
        """
        mapping of image plane to source plane coordinates
        WARNING: for multi lens plane computations and multi source planes, this computation can be slow and should be
        used as rarely as possible.

        :param x: image plane coordinate (angle)
        :param y: image plane coordinate (angle)
        :param kwargs_lens: lens model kwargs list
        :param index_source: int, index of source model
        :return: source plane coordinate corresponding to the source model of index idex_source
        """
        x_alpha, y_alpha = self._lensModel.alpha(x, y, kwargs_lens)
        scale_factor = self._deflection_scaling_list[index_source]
        x_source = x - x_alpha
        y_source = y - y_alpha
        return x_source, y_source

    def image_flux_joint(self, x, y, kwargs_lens, kwargs_source, k=None, k_lens=None):
        """

        :param x: coordinate in image plane
        :param y: coordinate in image plane
        :param kwargs_lens: lens model kwargs list
        :param kwargs_source: source model kwargs list
        :return: surface brightness of all joint light components at image position (x, y)
        """
        x_source, y_source = self._lensModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
        return self._lightModel.surface_brightness(x_source, y_source, kwargs_source, k=k)
