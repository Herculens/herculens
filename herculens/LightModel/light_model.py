# High-level interface to a light model
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


import numpy as np
import jax.numpy as jnp
# from functools import partial
# from jax import jit

from herculens.LightModel.light_model_base import LightModelBase
from herculens.Util import util


__all__ = ['LightModel']


class LightModel(LightModelBase):
    """Model extended surface brightness profiles of sources and lenses.

    Notes
    -----
    All profiles come with a surface_brightness parameterization (in units per
    square angle and independent of the pixel scale.) The parameter `amp` is
    the linear scaling parameter of surface brightness. Some profiles have
    a total_flux() method that gives the integral of the surface brightness
    for a given set of parameters.

    """
    def __init__(self, light_model_list, smoothing=0.001,
                 pixel_interpol='bilinear', kwargs_pixelated=None, 
                 shapelets_n_max=4, **kwargs):
        """Create a LightModel object."""
        self.profile_type_list = light_model_list
        super(LightModel, self).__init__(self.profile_type_list, smoothing=smoothing,
                                         pixel_interpol=pixel_interpol, 
                                         kwargs_pixelated=kwargs_pixelated,
                                         shapelets_n_max=shapelets_n_max, **kwargs)

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
        flux = jnp.zeros_like(x)
        bool_list = self._bool_list(k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                flux += func.function(x, y, **kwargs_list[i])
        return flux

    def spatial_derivatives(self, x, y, kwargs_list, k=None):
        """Spatial derivatives of the source flux at a given position (along x and y directions).

        Parameters
        ----------
        x, y : float or array_like
            Position coordinate(s) in arcsec relative to the image center.
        kwargs_list : list
            List of parameter dictionaries corresponding to each source model.
        k : int, optional
            Position index of a single source model component.

        """
        # x = jnp.array(x, dtype=float)
        # y = jnp.array(y, dtype=float)
        flux = jnp.zeros_like(x)
        bool_list = self._bool_list(k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                f_x_, f_y_ = func.derivatives(x, y, **kwargs_list[i])
                f_x += f_x_
                f_y += f_y_
        return f_x, f_y
