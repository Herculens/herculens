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
        self.func_list, self._pix_idx = self._load_model_instances(lens_model_list)
        self._num_func = len(self.func_list)
        self._model_list = lens_model_list
        self._kwargs_pixelated = kwargs_pixelated

    def _load_model_instances(self, lens_model_list):
        func_list = []
        imported_classes = {}
        pix_idx = None
        for idx, lens_type in enumerate(lens_model_list):
            # These models require a new instance per profile as certain pre-computations
            # are relevant per individual profile
            if lens_type in ['PIXELATED', 'PIXELATED_DIRAC']:
                mass_model_class = self._import_class(lens_type)
                pix_idx = idx
            else:
                if lens_type not in imported_classes.keys():
                    mass_model_class = self._import_class(lens_type)
                    imported_classes.update({lens_type: mass_model_class})
                else:
                    mass_model_class = imported_classes[lens_type]
            func_list.append(mass_model_class)
        return func_list, pix_idx

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
        return self._pix_idx is not None

    @property
    def pixel_grid_settings(self):
        return self._kwargs_pixelated

    def set_pixel_grid(self, pixel_grid):
        self.func_list[self.pixelated_index].set_pixel_grid(pixel_grid)

    @property
    def pixel_grid(self):
        if not self.has_pixels:
            return None
        return self.func_list[self.pixelated_index].pixel_grid

    @property
    def pixelated_index(self):
        # TODO: support multiple pixelated profiles
        return self._pix_idx

    @property
    def pixelated_coordinates(self):
        if not self.has_pixels:
            return None, None
        return self.func_list[idx].x_coords, self.func_list[idx].y_coords

    @property
    def pixelated_shape(self):
        if not self.has_pixels:
            return None
        x_coords, y_coords = self.pixelated_coordinates
        return (len(y_coords), len(x_coords))
