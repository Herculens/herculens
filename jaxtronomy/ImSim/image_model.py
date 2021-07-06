import functools
from jax import jit
import jax.numpy as jnp
from jaxtronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from jaxtronomy.ImSim.image2source_mapping import Image2SourceMapping
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel

__all__ = ['ImageModel']


class ImageModel(object):
    """Class that combines multiple modules to generate lensed images."""
    def __init__(self, data_class, psf_class, lens_model_class=None,
                 source_model_class=None, lens_light_model_class=None,
                 kwargs_numerics=None):
        """Generate lensed images from source, lens, and PSF models.

        Parameters
        ----------
        data_class : instance of `Data.pixel_grid.PixelGrid`
            Object describing and managing image grid coordinates.
        psf_class : instance of `Data.psf.PSF`
            Object describing and managing the point spread function.
        lens_model_class : instance of `LensModel.lens_model.LensModel`
            Object describing and managing the lens mass model(s).
        source_model_class : instance of `LightModel.light_model.LightModel`
            Object describing and managing the source light model(s).
        lens_light_model_class : instance of `LightModel.light_model.LightModel`
            Object describing and managing the lens light model(s).
        kwargs_numerics : dict
            Keyword arguments related to source light and PSF (super-)sampling.

        """
        self.Data = data_class
        self.PSF = psf_class
        self.PSF.set_pixel_size(self.Data.pixel_width)
        if kwargs_numerics is None:
            kwargs_numerics = {}
        self.ImageNumerics = NumericsSubFrame(pixel_grid=self.Data, psf=self.PSF,
                                              **kwargs_numerics)

        if lens_model_class is None:
            lens_model_class = LensModel(lens_model_list=[])
        self.LensModel = lens_model_class

        if source_model_class is None:
            source_model_class = LightModel(light_model_list=[])
        self.SourceModel = source_model_class

        if lens_light_model_class is None:
            lens_light_model_class = LightModel(light_model_list=[])
        self.LensLightModel = lens_light_model_class

        self.source_mapping = Image2SourceMapping(self.LensModel, self.SourceModel)

    def source_surface_brightness(self, kwargs_source, kwargs_lens=None,
                                  unconvolved=False, de_lensed=False, k=None):
        """Compute the source surface brightness on a grid.

        Parameters
        ----------
        kwargs_source : list of dict
            Keyword arguments specifying the superposed source light profiles.
        kwargs_lens : list of dict, optional
            Keyword arguments specifying the superposed lens mass profiles.
        unconvolved : bool, optional
            Whether or not the source light is convolved with the PSF.
            Default is False (meaning it is convolved).
        de_lensed : bool, optional
            Whether or not the source light is lensed. Default is False
            (meaning it is lensed).
        k : int, optional
            If given, compute only the source brightness corresponding to the
            k-th model.

        """
        if len(self.SourceModel.profile_type_list) == 0:
            return jnp.zeros((self.Data.num_pixel_axes))

        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        if de_lensed:
            source_light = self.SourceModel.surface_brightness(ra_grid, dec_grid,
                                                               kwargs_source, k=k)
        else:
            source_light = self.source_mapping.image_flux_joint(ra_grid, dec_grid,
                                                                kwargs_lens,
                                                                kwargs_source, k=k)
        return self.ImageNumerics.re_size_convolve(source_light, unconvolved)

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """Compute the lens surface brightness on a grid.

        Parameters
        ----------
        kwargs_lens_light : list of dict
            Keyword arguments specifying the superposed lens light profiles.
        unconvolved : bool, optional
            Whether or not the lens light is convolved with the PSF.
            Default is False (meaning it is convolved).
        k : int, optional
            If given, compute only the lens brightness corresponding to the
            k-th model.

        """
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        lens_light = self.LensLightModel.surface_brightness(ra_grid, dec_grid,
                                                            kwargs_lens_light, k=k)
        return self.ImageNumerics.re_size_convolve(lens_light, unconvolved)

    @functools.partial(jit, static_argnums=(0,))
    def image(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
              unconvolved=False, source_add=True, lens_light_add=True):
        """Generate a lensing image based on specified input model parameters.

        Parameters
        ----------
        kwargs_lens : list of dict
            Keyword arguments specifying the superposed lens mass profiles.
        kwargs_source : list of dict
            Keyword arguments specifying the superposed source light profiles.
        kwargs_lens_light : list of dict
            Keyword arguments specifying the superposed lens light profiles.
        unconvolved : bool, optional
            Whether or not the lens and source light distributions are convolved
            with the PSF. Default is False (meaning it is convolved).
        source_add : bool, optional
            Whether to include the source light in the final image. Default is True.
        lens_light_add : bool, optional
            Whether to include the lens light in the final image. Default is True.

        """
        model = jnp.zeros((self.Data.num_pixel_axes))
        if source_add:
            model += self.source_surface_brightness(kwargs_source, kwargs_lens,
                                                    unconvolved=unconvolved)
        if lens_light_add:
            model += self.lens_surface_brightness(kwargs_lens_light,
                                                  unconvolved=unconvolved)
        return model
