# Copyright (c) 2022, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the base_ps module from lenstronomy (version 1.11.0)

__author__ = 'sibirrer', 'austinpeel'

from herculens.MassModel.LensEquation.solver import LensEquationSolver

__all__ = ['PointSourceBase']

class PointSourceModelBase(object):
    """Base class for a point source model."""
    def __init__(self, mass_model=None):
        self._mass_model = mass_model
        if self._mass_model is None:
            self._solver = None
        else:
            self._solver = LensEquationSolver(self._mass_model)

    def image_positions(self, kwargs_point_source, **kwargs):
        """Positions of a lensed point source in the image plane.

        :param kwargs_point_source: keyword arguments of the point source
        :return: array of x, y image positions
        """
        raise ValueError('image_positions has not been defined.')

    def source_position(self, kwargs_point_source, **kwargs):
        """Position of a point source in the source plane.

        :param kwargs_point_source: keyword arguments of the point source
        :return: array of x, y source positions
        """
        raise ValueError('source_position has not been defined.')

    def image_amplitudes(self, kwargs_point_source, **kwargs):
        """Amplitudes of point sources in the image plane.

        :param kwargs_point_source: keyword arguments of the point source
        :return: array of amplitudes
        """
        raise ValueError('image_amplitudes has not been defined.')

    def source_amplitude(self, kwargs_point_source, **kwargs):
        """Amplitude of a point source in the source plane.

        :param kwargs_point_source: keyword arguments of the point source
        :return: array of amplitudes
        """
        raise ValueError('source amplitudes has not been defined.')

    def update_mass_model(self, mass_model):
        """Update the mass model for ray shooting and solving the lens equation.

        :param mass_model: instance of `herculens.MassModel.mass_model.MassModel`
        """
        self._mass_model = mass_model
        if self._mass_model is None:
            self._solver = None
        else:
            self._solver = LensEquationSolver(self._mass_model)
