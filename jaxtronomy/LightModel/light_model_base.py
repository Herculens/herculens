import jax.numpy as jnp
from jaxtronomy.LightModel.Profiles import sersic, pixelated, uniform
from jaxtronomy.Util.util import convert_bool_list

__all__ = ['LightModelBase']


_SUPPORTED_MODELS = ['SERSIC', 'SERSIC_ELLIPSE', 'CORE_SERSIC', 'UNIFORM', 'PIXELATED']


class LightModelBase(object):
    """Base class for source and lens light models."""
    def __init__(self, light_model_list, smoothing=0.001,
                 pixel_x_coords=None, pixel_y_coords=None, pixel_interpol='bilinear'):
        """Create a LightModelBase object.

        Parameters
        ----------
        light_model_list : list of str
            Light model types.
        smoothing : float
            Smoothing factor for some models (deprecated).

        """
        self.profile_type_list = light_model_list
        self._pixel_x_coords, self._pixel_y_coords = pixel_x_coords, pixel_y_coords
        func_list = []
        for profile_type in light_model_list:
            if profile_type == 'SERSIC':
                func_list.append(sersic.Sersic(smoothing))
            elif profile_type == 'SERSIC_ELLIPSE':
                func_list.append(sersic.SersicElliptic(smoothing))
            elif profile_type == 'CORE_SERSIC':
                func_list.append(sersic.CoreSersic(smoothing))
            elif profile_type == 'UNIFORM':
                func_list.append(uniform.Uniform())
            elif profile_type == 'PIXELATED':
                func_list.append(pixelated.Pixelated(self._pixel_x_coords, self._pixel_y_coords, 
                                                     method=pixel_interpol))
            else:
                err_msg = (f"No light model of type {profile_type} found. " +
                           f"Supported types are: {_SUPPORTED_MODELS}")
                raise ValueError(err_msg)
        self.func_list = func_list
        self._num_func = len(self.func_list)

    def surface_brightness(self, x, y, kwargs_list, k=None):
        """Total source flux at a given position.

        Parameters
        ----------
        x, y : float or array_like
            Position coordinate(s) in arcsec relative to the image center.
        kwargs_list : list
            List of parameter dictionaries corresponding to each source model.
        k : int, optional
            Position index of a single source model component.

        """
        x = jnp.array(x, dtype=float)
        y = jnp.array(y, dtype=float)
        flux = jnp.zeros_like(x)
        bool_list = convert_bool_list(self._num_func, k=k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                flux += func.function(x, y, **kwargs_list[i])
        return flux

    @property
    def pixelated_index(self):
        if not hasattr(self, '_pix_idx'):
            try:
                self._pix_idx = self.profile_type_list.index('PIXELATED')
            except ValueError:
                self._pix_idx = None
        return self._pix_idx

    @property
    def pixelated_shape(self):
        if not hasattr(self, '_pix_shape'):
            idx = self.pixelated_index
            if idx is not None:
                if self._pixel_x_coords is None or self._pixel_y_coords is None:
                    raise RuntimeError("There is a 'PIXELATED' light profile but "
                                       "no coordinate arrays have been provided")
                self._pix_shape = (len(self._pixel_x_coords), len(self._pixel_y_coords))
            else:
                self._pix_shape = None
        return self._pix_shape
