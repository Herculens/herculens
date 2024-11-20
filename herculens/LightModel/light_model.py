# High-level interface to a light model
# 
# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'

from functools import partial
import jax.numpy as jnp

from herculens.LightModel.light_model_base import LightModelBase


__all__ = ['LightModel']


def function_static_single(
        x, y, function,
    ):
    return partial(function, x, y)


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
    def __init__(self, profile_list, verbose=False, **kwargs):
        """Create a LightModel object.

        Parameters
        ----------
        profile_list : list of strings or profile instances
            List of light profiles.
        kwargs_pixelated : dictionary for settings related to PIXELATED profiles.
        """
        if not isinstance(profile_list, (list, tuple)):
            # useful when using a single profile
            profile_list = [profile_list]
        self.profile_type_list = profile_list
        super(LightModel, self).__init__(self.profile_type_list, **kwargs)
        self._single_profile_mode = False
        if len(self.profile_type_list) > 0:
            first_profile = self.profile_type_list[0]
            self._single_profile_mode = (
                all(p is first_profile for p in self.profile_type_list)
            )
            if verbose is True and self._single_profile_mode:
                print("Single profile mode in LightModel.")

    def surface_brightness(self, x, y, kwargs, k=None):
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
        # x = np.array(x, dtype=float)
        # y = np.array(y, dtype=float)
        if isinstance(k, int):
            return self.func_list[k].function(x, y, **kwargs[k])
        elif self._single_profile_mode:
            return self._surf_bright_single(x, y, kwargs, k=k)
        else:
            return self._surf_bright_loop(x, y, kwargs, k=k)
        
    def _surf_bright_single(self, x, y, kwargs, k=None):
        if k is not None:
            raise NotImplementedError   # TODO: implement case with k not None
        func = function_static_single(x, y, self.func_list[0].function)
        return jnp.sum(
            jnp.array([
                func(**kwargs[i]) for i in range(self._num_func)
            ]),
            axis=0,
        )

    def _surf_bright_loop(self, x, y, kwargs_list, k=None,
                          pixels_x_coord=None, pixels_y_coord=None):
        flux = jnp.zeros_like(x)
        bool_list = self._bool_list(k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                if i == self.pixelated_index:
                    flux += func.function(x, y, 
                                          pixels_x_coord=pixels_x_coord, 
                                          pixels_y_coord=pixels_y_coord, 
                                          **kwargs_list[i])
                else:
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
        f_x, f_y = jnp.zeros_like(x), jnp.zeros_like(x)
        bool_list = self._bool_list(k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                f_x_, f_y_ = func.derivatives(x, y, **kwargs_list[i])
                f_x += f_x_
                f_y += f_y_
        return f_x, f_y
