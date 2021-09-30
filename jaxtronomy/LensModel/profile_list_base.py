from jaxtronomy.LensModel.Profiles import (gaussian_potential,
                                           shear, sie, nie, epl, pixelated)
from jaxtronomy.Util.util import convert_bool_list

__all__ = ['ProfileListBase']

SUPPORTED_MODELS = ['EPL', 'NIE', 'SIE', 'GAUSSIAN', 'SHEAR', 'SHEAR_GAMMA_PSI', 'PIXELATED']


class ProfileListBase(object):
    """Base class for managing lens models in single- or multi-plane lensing."""
    def __init__(self, lens_model_list, lens_redshift_list=None, kwargs_pixelated={}):
        """Create a ProfileListBase object.

        Parameters
        ----------
        lens_model_list : list of str
            Lens model profile types.
        lens_redshift_list : list of float, optional
            Lens redshifts corresponding to the profiles in `lens_model_list`.

        """
        self.func_list = self._load_model_instances(lens_model_list, lens_redshift_list)
        self._num_func = len(self.func_list)
        self._model_list = lens_model_list
        self._kwargs_pixelated = kwargs_pixelated

    def _load_model_instances(self, lens_model_list, lens_redshift_list=None):
        if lens_redshift_list is None:
            lens_redshift_list = [None] * len(lens_model_list)
        func_list = []
        imported_classes = {}
        for lens_type in lens_model_list:
            # These models require a new instance per profile as certain pre-computations
            # are relevant per individual profile
            if lens_type in ['PIXELATED']:
                lensmodel_class = self._import_class(lens_type)
            else:
                if lens_type not in imported_classes.keys():
                    lensmodel_class = self._import_class(lens_type)
                    imported_classes.update({lens_type: lensmodel_class})
                else:
                    lensmodel_class = imported_classes[lens_type]
            func_list.append(lensmodel_class)
        return func_list

    def _import_class(self, lens_type):
        """Get the lens profile class of the corresponding type."""
        if lens_type == 'GAUSSIAN':
            return gaussian_potential.Gaussian()
        elif lens_type == 'SHEAR':
            return shear.Shear()
        elif lens_type == 'SHEAR_GAMMA_PSI':
            return shear.ShearGammaPsi()
        elif lens_type == 'NIE':
            return nie.NIE()
        elif lens_type == 'SIE':
            return sie.SIE()
        elif lens_type == 'EPL':
            return epl.EPL()
        elif lens_type == 'PIXELATED':
            return pixelated.PixelatedPotential()
        else:
            err_msg = (f"{lens_type} is not a valid lens model. " +
                       f"Supported types are {SUPPORTED_MODELS}")
            raise ValueError(err_msg)

    def _bool_list(self, k=None):
        """See `Util.util.convert_bool_list`."""
        return convert_bool_list(n=self._num_func, k=k)

    def set_static(self, kwargs_list):
        """Pre-compute lensing quantities for faster (but fixed) execution."""
        for kwargs, func in zip(kwargs, self.func_list):
            func.set_static(**kwargs)
        return kwargs_list

    def set_dynamic(self):
        """Free the cache of pre-computed quantities from `set_static`.

        This mode recomputes lensing quantities each time a method is called.
        This is the default mode if `set_static` has not been called.

        """
        for func in self.func_list:
            func.set_dynamic()

    @property
    def has_pixels(self):
        return ('PIXELATED' in self._model_list)

    @property
    def pixel_grid_settings(self):
        return self._kwargs_pixelated

    def set_pixel_grid(self, pixel_axes):
        for i, func in enumerate(self.func_list):
            if self._model_list[i] == 'PIXELATED':
                func.set_data_pixel_grid(pixel_axes)

    @property
    def pixelated_index(self):
        if not hasattr(self, '_pix_idx'):
            try:
                self._pix_idx = self._model_list.index('PIXELATED')
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
