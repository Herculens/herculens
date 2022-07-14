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
    def __init__(self, light_model_list, deflection_scaling_list=None,
                 source_redshift_list=None, smoothing=0.001,
                 pixel_interpol='bilinear', pixel_allow_extrapolation=False, kwargs_pixelated={}):
        """Create a LightModel object."""
        super(LightModel, self).__init__(light_model_list, smoothing, 
                                         pixel_interpol=pixel_interpol, 
                                         pixel_allow_extrapolation=pixel_allow_extrapolation,
                                         kwargs_pixelated=kwargs_pixelated)
