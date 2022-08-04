# High-level interface to a light model
# 
# Copyright (c) 2021, herculens developers and contributors
# Copyright (c) 2018, Simon Birrer & lenstronomy contributors
# based on the LightModel module from lenstronomy (version 1.9.3)

__author__ = 'sibirrer', 'austinpeel', 'aymgal'


from herculens.LightModel.light_model_base import LightModelBase

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
    def __init__(self, light_model_list, **kwargs):
        """Create a LightModel object."""
        super(LightModel, self).__init__(light_model_list, **kwargs)
