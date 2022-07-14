# Describes a mass model, as a list of mass profiles
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LensModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


from herculens.MassModel.Profiles import (gaussian_potential, point_mass, multipole,
                                           shear, sie, sis, nie, epl, pixelated)
from herculens.Util.util import convert_bool_list

__all__ = ['MassProfileBase']

SUPPORTED_MODELS = [
    'EPL', 'NIE', 'SIE', 'SIS', 'GAUSSIAN', 'POINT_MASS', 
    'SHEAR', 'SHEAR_GAMMA_PSI', 'MULTIPOLE',
    'PIXELATED', 'PIXELATED_DIRAC',
]


class MassProfileBase(object):
    """Base class for managing lens models in single- or multi-plane lensing."""
    def __init__(self, lens_model_list, kwargs_pixelated={}):
        """Create a MassProfileBase object.

        Parameters
        ----------
        lens_model_list : list of str
            Lens model profile types.

        """
        self.func_list = self._load_model_instances(lens_model_list)
        self._num_func = len(self.func_list)
        self._model_list = lens_model_list
        self._kwargs_pixelated = kwargs_pixelated

    def _load_model_instances(self, lens_model_list):
        func_list = []
        imported_classes = {}
        for lens_type in lens_model_list:
            # These models require a new instance per profile as certain pre-computations
            # are relevant per individual profile
            if lens_type in ['PIXELATED', 'PIXELATED_DIRAC']:
                mass_model_class = self._import_class(lens_type)
            else:
                if lens_type not in imported_classes.keys():
                    mass_model_class = self._import_class(lens_type)
                    imported_classes.update({lens_type: mass_model_class})
                else:
                    mass_model_class = imported_classes[lens_type]
            func_list.append(mass_model_class)
        return func_list

    @staticmethod
    def _import_class(lens_type):
        """Get the lens profile class of the corresponding type."""
        if lens_type == 'GAUSSIAN':
            return gaussian_potential.Gaussian()
        elif lens_type == 'SHEAR':
            return shear.Shear()
        elif lens_type == 'SHEAR_GAMMA_PSI':
            return shear.ShearGammaPsi()
        elif lens_type == 'POINT_MASS':
            return point_mass.PointMass()
        elif lens_type == 'NIE':
            return nie.NIE()
        elif lens_type == 'SIE':
            return sie.SIE()
        elif lens_type == 'SIS':
            return sis.SIS()
        elif lens_type == 'EPL':
            return epl.EPL()
        elif lens_type == 'MULTIPOLE':
            return multipole.Multipole()
        elif lens_type == 'PIXELATED':
            return pixelated.PixelatedPotential()
        elif lens_type == 'PIXELATED_DIRAC':
            return pixelated.PixelatedPotentialDirac()
        else:
            err_msg = (f"{lens_type} is not a valid lens model. " +
                       f"Supported types are {SUPPORTED_MODELS}")
            raise ValueError(err_msg)

    def _bool_list(self, k=None):
        """See `Util.util.convert_bool_list`."""
        return convert_bool_list(n=self._num_func, k=k)

    @property
    def has_pixels(self):
        return ('PIXELATED' in self._model_list) or ('PIXELATED_DIRAC' in self._model_list)

    @property
    def pixel_grid_settings(self):
        return self._kwargs_pixelated

    def set_pixel_grid(self, pixel_axes):
        for i, func in enumerate(self.func_list):
            if self._model_list[i] in ['PIXELATED', 'PIXELATED_DIRAC']:
                func.set_data_pixel_grid(pixel_axes)

    @property
    def pixelated_index(self):
        if not hasattr(self, '_pix_idx'):
            try:
                self._pix_idx = self._model_list.index('PIXELATED')
            except ValueError:
                try:
                    self._pix_idx = self._model_list.index('PIXELATED_DIRAC')
                except ValueError:
                    self._pix_idx = None
        return self._pix_idx

    @property
    def pixelated_coordinates(self):
        idx = self.pixelated_index
        if idx is None:
            return None, None
        return self.func_list[idx].x_coords, self.func_list[idx].y_coords

    @property
    def pixelated_shape(self):
        x_coords, y_coords = self.pixelated_coordinates
        if x_coords is None:
            return None
        else:
            return (len(y_coords), len(x_coords))
