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
        self._repeated_profile_mode = False
        self._single_profile_mode = len(self.profile_type_list) == 1
        if len(self.profile_type_list) > 0:
            first_profile = self.profile_type_list[0]
            self._repeated_profile_mode = (
                all(p is first_profile for p in self.profile_type_list)
            )
            if verbose is True and self._repeated_profile_mode:
                print("All LightModel profiles are identical.")

    def surface_brightness(self, x, y, kwargs, k=None,
                           pixels_x_coord=None, pixels_y_coord=None,
                           return_as_list=False):
            """Total source flux at a given position.

            Parameters
            ----------
            x : float or array_like
                Position coordinate(s) in arcsec relative to the image center.
            y : float or array_like
                Position coordinate(s) in arcsec relative to the image center.
            kwargs : list
                List of parameter dictionaries corresponding to each source model.
            k : int, optional
                Position index of a single source model component.
            pixels_x_coord : array_like, optional
                x-coordinates of the pixelated light profile (if any).
            pixels_y_coord : array_like, optional
                y-coordinates of the pixelated light profile (if any).
            return_as_list : bool, optional
                If True, return the flux of each profile separately.

            Returns
            -------
            float or array_like
                Total source flux at the given position(s).

            """
            # x = np.array(x, dtype=float)
            # y = np.array(y, dtype=float)
            if isinstance(k, int):
                return self._surf_bright_single(x, y, kwargs, k=k,
                                                pixels_x_coord=pixels_x_coord,
                                                pixels_y_coord=pixels_y_coord,
                                                return_as_list=return_as_list)
            elif self._single_profile_mode:
                return self._surf_bright_single(x, y, kwargs, k=0,
                                                pixels_x_coord=pixels_x_coord,
                                                pixels_y_coord=pixels_y_coord,
                                                return_as_list=return_as_list)
            elif self._repeated_profile_mode:
                return self._surf_bright_repeated(x, y, kwargs, k=k,
                                                  pixels_x_coord=pixels_x_coord,
                                                  pixels_y_coord=pixels_y_coord,
                                                  return_as_list=return_as_list)
            else:
                return self._surf_bright_loop(x, y, kwargs, k=k,
                                              pixels_x_coord=pixels_x_coord,
                                              pixels_y_coord=pixels_y_coord,
                                              return_as_list=return_as_list)
            
    def _surf_bright_single(self, x, y, kwargs, k=None,
                            pixels_x_coord=None, pixels_y_coord=None,
                            return_as_list=False):
        if k == self.pixelated_index:
            flux = self.func_list[k].function(
                x, y, **kwargs[k],
                pixels_x_coord=pixels_x_coord, 
                pixels_y_coord=pixels_y_coord,
            )
        else:
            flux = self.func_list[k].function(x, y, **kwargs[k])
        if return_as_list:
            return [flux]
        return flux
        
    def _surf_bright_repeated(self, x, y, kwargs, k=None,
                              pixels_x_coord=None, pixels_y_coord=None,
                              return_as_list=False):
        if k is not None:
            raise NotImplementedError("Repeated profile mode not implemented "
                                      "specific profile k.")
        func = function_static_single(x, y, self.func_list[0].function)
        flux_list = [
            func(**kwargs[i]) for i in range(self._num_func)
        ]
        if return_as_list:
            return flux_list
        return jnp.sum(jnp.array(flux_list),axis=0)

    def _surf_bright_loop(self, x, y, kwargs_list, k=None,
                          pixels_x_coord=None, pixels_y_coord=None,
                          return_as_list=False):
        if return_as_list:
            flux = []
        else:
            flux = jnp.zeros_like(x)
        bool_list = self._bool_list(k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                if i == self.pixelated_index:
                    flux_i = func.function(x, y, 
                                           pixels_x_coord=pixels_x_coord, 
                                           pixels_y_coord=pixels_y_coord, 
                                           **kwargs_list[i])
                else:
                    flux_i = func.function(x, y, **kwargs_list[i])
                if return_as_list:
                    flux.append(flux_i)
                else:
                    flux += flux_i
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
