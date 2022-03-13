#Basic imports
import numpy as np
from copy import deepcopy

#JAX
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

#Herculens
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.LensModel.lens_model import LensModel
from herculens.LensImage.lens_image import LensImage
from herculens.Parameters.parameters import Parameters

#Default models of source galaxy and lens galaxy
default_source_light_model_list = ['SERSIC_ELLIPSE']
default_lens_mass_model_list = ['SIE', 'SHEAR']
default_lens_light_model_list = []

# Default model parameters of source galaxy and lens galaxy
default_unperturbed_model_kwargs={'kwargs_lens': [{'theta_E': 1.6, 'e1': 0.15, 'e2': -0.04, 'center_x': 0.0, 'center_y': 0.0},
                                                  {'gamma1': -0.01, 'gamma2': 0.03, 'ra_0': 0.0, 'dec_0': 0.0}],
                                  'kwargs_source': [{'amp': 10.0, 'R_sersic': 1.2, 'n_sersic': 1.5, 'center_x': 0.4, 'center_y': 0.15,'e1':0.07,'e2':-0.1}],
                                  'kwargs_lens_light': [{}]}

# Priors on default parameters from this article Park et al. 2021 Table 1 (Partially)
default_kwargs_prior = {'kwargs_lens': [{'theta_E': ['uniform', 1., 2.], 'e1': ['gaussian', 0, 0.2], 'e2': ['gaussian', 0, 0.2],'center_x': ['gaussian', 0, 0.102], 'center_y': ['gaussian', 0, 0.102]},
                                        {'gamma1': ['uniform', -0.5, 0.5], 'gamma2': ['uniform', -0.5, 0.5]}],
                        'kwargs_source': [{'amp': ['uniform',5.0,20.0], 'R_sersic': ['uniform',1e-3,5.], 'n_sersic': ['uniform',1e-3,4.0],
                                           'center_x': ['uniform', -1.0, 1.0], 'center_y': ['uniform', -1.0, 1.0],'e1': ['gaussian', 0, 0.2], 'e2': ['gaussian', 0, 0.2]}]}

# These are just mean values of the priors. Some have minor offsets from zero for smooth fitting start
default_kwargs_init = {'kwargs_lens': [{'theta_E': 1.5,'e1': 1e-3,'e2': 1e-3,'center_x': 1e-3,'center_y': 1e-3},
                                       {'gamma1': 1e-3, 'gamma2': 1e-3, 'ra_0': 0.0, 'dec_0': 0.0}],
                       'kwargs_source': [{'amp': 5.0,'R_sersic': 2.5,'n_sersic': 2.,'center_x': 0.,'center_y': 0.,'e1': 1e-3,'e2': 1e-3}]}

# The quantities that remain fixed during source-lens model fitting procedure
default_kwargs_fixed = {
    'kwargs_lens': [{}, {'ra_0': 0., 'dec_0': 0.}],  # fix origin of the external shear profile
    'kwargs_source': [{}]}  # fix all source parameters


def check_model(model, kwargs):
    """
    Check that model is list of stings
    Check that kwargs is list of dictionaries with string keys and numeric values
    Check that their length are the same
    Parameters
    ----------
    model: [str,str...] or []
        Names of source/lens models
    kwargs: [{str:float,...},...] or [{}]
        Parameters of source/lens model

    Returns
    -------
    True or exception

    """
    # Empty model
    if (model == [] or model == [{}]) and (kwargs == [{}] or kwargs == []):
        return True

    #Check model
    try:
        if not (set(map(type, model)) == {str}):
            raise
    except:
        raise ValueError('model should be list/array of strings')

    #Check kwargs
    try:
        key_types = [set(map(type, model_dict.keys())) for model_dict in kwargs]
        value_types = [set(map(type, model_dict.values())) for model_dict in kwargs]
        are_str_keys = set().union(*key_types) == {str}
        are_numeric_values = set().union(*value_types).issubset({float, int})
        if not are_str_keys & are_numeric_values:
            raise
    except:
        raise ValueError('kwargs should be a list/array of dictionaries with string keys and float values')

    #Check number of models and number of their parameters
    if not len(model) == len(kwargs):
        raise ValueError('Model list and kwargs list have different length')

    return True

class Surface_brightness_class:
    """A class that generates mock surface brightness of the gravitational lens
    for predefined source galaxy sb, lens galaxy sb, lens galaxy gravitational potential
    and for varying galaxy satellites gravitational potential"""

    def __init__(self, pixel_number: int, pixel_scale: float, PSF_class: float, bkg_noise_sigma: float, exposure_time: float, supersampling_factor=None,
                 source_light_model_list=None, kwargs_source_light=None,
                 lens_mass_model_list=None, kwargs_lens_mass=None,
                 lens_light_model_list=None, kwargs_lens_light=None,
                 annulus_mask_borders=None):
        """
        Initialize model for given observation conditions and unperturbed source-lens setup
        Parameters
        ----------
        pixel_number: int (even)
            Lens plane grid is a square with side of 'pixel_number' pixels
        pixel_scale: float
            Resolution 'arcsec/pixel'
        PSF_class: object herculens.Instrument.psf.PSF
            class defining PSF
        bkg_noise_sigma: float
            Std of background noise
        exposure_time: float
            Exposure time for Poisson noise in seconds
        supersampling_factor: int
            factor of higher resolution sub-pixel sampling of surface brightness
        source_light_model_list: [str,str,...] or []
            Names of models used to define light in the source plane
        kwargs_source_light: [{str:float,...},...] or [{}]
            Parameters of models used to define light in the source plane
        lens_mass_model_list: [str,str,...] or []
            Names of models used to define mass in the lens plane
        kwargs_lens_mass: [{str:float,...},...] or [{}]
            Parameters of models used to define mass in the lens plane
        lens_light_model_list: [str,str,...] or []
            Names of models used to define light in the lens plane
        kwargs_lens_light: [{str:float,...},...] or [{}]
            Parameters of models used to define light in the lens plane
        annulus_mask_borders: [float,float]
            Inner and Outer borders of the mask covering Einstien ring in units of arcsec
        """

        # Even shaped square grid facilitates geometrical trnasformations, but should be generalised in the future
        assert (type(pixel_number)==int) and (pixel_number%2==0)

        #Size of grid and resolution
        self.pixel_number=pixel_number
        self.pixel_scale=pixel_scale

        # Set up PixelGrid
        half_size = pixel_number * pixel_scale / 2
        ra_at_xy_0 = dec_at_xy_0 = -half_size + pixel_scale / 2
        transform_pix2angle = pixel_scale * np.eye(2)
        kwargs_pixel = {'nx': pixel_number, 'ny': pixel_number,
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': transform_pix2angle}
        self.pixel_grid=PixelGrid(**kwargs_pixel)

        #Pixelisation of source and lens plane
        if supersampling_factor is None:
            supersampling_factor = 1
        self.kwargs_numerics = {'supersampling_factor': supersampling_factor}

        #PreInitialize unperturbed source-lens models and their parameters
        self.kwargs_unperturbed_model=default_unperturbed_model_kwargs
        self.source_light_model_list=default_source_light_model_list
        self.lens_mass_model_list=default_lens_mass_model_list
        self.lens_light_model_list=default_lens_light_model_list

        #Set up Source light parameters if proper model and parameters are given
        if (source_light_model_list is not None)  and (kwargs_source_light is not None)\
            and check_model(source_light_model_list,kwargs_source_light):
            self.source_light_model_list = source_light_model_list
            self.kwargs_unperturbed_model['kwargs_source'] = kwargs_source_light

        #Set up Lens mass parameters if proper model and parameters are given
        if (lens_mass_model_list is not None) and (kwargs_lens_mass is not None)\
            and check_model(lens_mass_model_list,kwargs_lens_mass):
            self.lens_mass_model_list=lens_mass_model_list
            self.kwargs_unperturbed_model['kwargs_lens']=kwargs_lens_mass


        #Set up Lens light parameters if proper model and parameters are given
        if (lens_light_model_list is not None) and (kwargs_lens_light is not None)\
            and check_model(lens_light_model_list,kwargs_lens_light):
            self.lens_light_model_list=lens_light_model_list
            self.kwargs_unperturbed_model['kwargs_lens_light']=kwargs_lens_light

        # Mask covering Einstein ring
        if annulus_mask_borders is None:
            annulus_mask_borders = [0.5, 3]

        radius = np.hypot(*self.pixel_grid.pixel_coordinates)
        rmin, rmax = annulus_mask_borders
        self.annulus_mask = ((radius >= rmin) & (radius <= rmax))

        # There is no sense to consider wavevectors referring to sizes biger than the thickness of the masked region
        full_freq_vector = np.fft.fftshift(np.fft.fftfreq(self.pixel_number, self.pixel_scale))[self.pixel_number // 2:]
        # Index from which to consider wavenumbers
        self.init_freq_index = np.where(full_freq_vector > 1 / (rmax - rmin))[0][0]
        # Relevant wavenumbers
        self.frequencies = full_freq_vector[self.init_freq_index:]

        #Setup all the observation conditions
        self.PSF_class=PSF_class
        self.exposure_time=exposure_time
        self.bkg_noise_sigma = bkg_noise_sigma
        #Lens images for noiseless observations
        LensImage_perturbed_noiseless, LensImage_unperturbed_noiseless = self.get_LensImages(1e-10,None)
        self.LensImage_unperturbed_noiseless=LensImage_unperturbed_noiseless
        self.LensImage_perturbed_noiseless=LensImage_perturbed_noiseless
        # Evaluate noise level from Peak-SNR of noiseless image
        Image_unperturbed_noiseless=LensImage_unperturbed_noiseless.simulation(**self.kwargs_unperturbed_model)
        #self.bkg_noise_sigma=Image_unperturbed_noiseless.max()/SNR
        #Poisson+Background noise
        self.noise_var=np.abs(Image_unperturbed_noiseless)/exposure_time+self.bkg_noise_sigma**2

        # Lens images for noisy setup
        LensImage_perturbed_noisy, LensImage_unperturbed_noisy = self.get_LensImages(self.bkg_noise_sigma,exposure_time)
        self.LensImage_perturbed_noisy=LensImage_perturbed_noisy
        self.LensImage_unperturbed_noisy=LensImage_unperturbed_noisy

    @property
    def grid_coordinates(self):
        xgrid, ygrid = self.pixel_grid.pixel_coordinates
        x_coords = xgrid[0, :]
        y_coords = ygrid[:, 0]
        return x_coords,y_coords

    @property
    def unperturbed_image_getter(self):
        def simulate_unperturbed_image(model_kwargs, Noise_flag=True, noise_seed=42):
            '''
            Function that simulates observation for mock source-lens parameters assuming no galaxy satellites
            Parameters
            ----------
            model_kwargs: {'kwargs_lens':[{str:float,...},...],...], 'kwargs_source': ...,'kwargs_lens_light':...}
                Parameters of source-lens model assuming no galaxy satellites
            Noise_flag: bool
                Presence of noise
            noise_seed: unit32
                seed for noise generation

            Returns
            -------
            image: jnp.ndarray  real (pixel_number,pixel_number)
                Mock observation for given source-lens galaxies setup assumign no galaxy satellites
            '''

            if Noise_flag:
                LensImage_unperturbed = self.LensImage_unperturbed_noisy
            else:
                LensImage_unperturbed = self.LensImage_unperturbed_noiseless

            return LensImage_unperturbed.simulation(noise_seed=noise_seed, **model_kwargs)

        return simulate_unperturbed_image

    @property
    def perturbed_image_getter(self):
        def simulate_perturbed_image(GRF_potential, Noise_flag=True, noise_seed=42):
            '''
            Function that simulates observation for mock GRF potential inhomogeneities
            Parameters
            ----------
            GRF_potential: jnp.ndarray
                Potential of GRF perturbations as array of shape (pixel_number,pixel_number)
            Noise_flag: bool
                Presence of noise
            noise_seed: unit32
                seed for noise generation

            Returns
            -------
            image: jnp.ndarray  real (pixel_number,pixel_number)
                Mock observation for given GRF potential of galaxy satellites
            '''

            kwargs_lens = self.kwargs_unperturbed_model['kwargs_lens'] + [{'pixels': GRF_potential}]

            if Noise_flag:
                LensImage_perturbed = self.LensImage_perturbed_noisy
            else:
                LensImage_perturbed = self.LensImage_perturbed_noiseless

            return LensImage_perturbed.simulation(kwargs_lens=kwargs_lens,
                                                  kwargs_source=self.kwargs_unperturbed_model['kwargs_source'],
                                                  kwargs_lens_light=self.kwargs_unperturbed_model['kwargs_lens_light'],
                                                  noise_seed=noise_seed)

        return simulate_perturbed_image

    def get_LensImages(self,noise_sigma,exposure_time):
        """

        Parameters
        ----------
        noise_sigma: float
            Std of background noise in units of flux
        exposure_time: float
            Exposure time for Poisson noise in seconds

        Returns
        -------
        herculens.LensImage.lens_image.LensImage,herculens.LensImage.lens_image.LensImage
            Class for perturbed lens simulationa and class for unperturbed lens simulation correspondingly
        """
        #PSF_class = PSF(**{'psf_type': 'GAUSSIAN', 'fwhm': PSF_FWHM})
        pixel_number, _ = self.pixel_grid.num_pixel_axes
        Noise_class = Noise(pixel_number, pixel_number,
                            **{'background_rms': noise_sigma, 'exposure_time': exposure_time})

        # Unperturbed lens
        unperturbed_kwargs_for_LensImage={'grid_class':self.pixel_grid,'psf_class':self.PSF_class,'noise_class':Noise_class,
                                 'lens_model_class':LensModel(self.lens_mass_model_list),'source_model_class':LightModel(self.source_light_model_list),
                                 'lens_light_model_class':LightModel(self.lens_light_model_list),'kwargs_numerics':self.kwargs_numerics}

        unperturbed_lens_image = LensImage(**unperturbed_kwargs_for_LensImage)

        # Perturbed lens, where we add PIXELATED potential to lens_model to describe galaxy satellites
        perturbed_kwargs_for_LensImage=deepcopy(unperturbed_kwargs_for_LensImage)
        perturbed_kwargs_for_LensImage['lens_model_class']=LensModel(self.lens_mass_model_list+['PIXELATED'])

        perturbed_lens_image = LensImage(**perturbed_kwargs_for_LensImage)

        return perturbed_lens_image,unperturbed_lens_image

    def parameters(self,kwargs_prior=None,kwargs_init=None,kwargs_fixed=None):
        """
        Return the Parameters class needed to define fitting procedure
        Parameters
        ----------
        kwargs_prior: {[{str:[str,float,float,...]},...],...}
            Priors on the parameters, e.g. {'kwargs_lens':[{'theta_E':['uniform', 1., 2.]},...],...}
        kwargs_init: {[{str:float,...},...],...}
            Values from which the fitting starts, e.g. {'kwargs_lens': [{'theta_E': 1.5,...},...],...}
        kwargs_fixed: {[{str:float,...},...],...}
            Quantities that are fixed during fitting, e.g. {'kwargs_lens': [{}, {'ra_0': 0., 'dec_0': 0.}],...}

        Returns
        -------
            Parameters: herculens.Parameters.parameters.Parameters
        """

        if kwargs_prior is None:
            kwargs_prior=default_kwargs_prior

        if kwargs_init is None:
            kwargs_init=default_kwargs_init

        if kwargs_fixed is None:
            kwargs_fixed=default_kwargs_fixed

        return Parameters(self.LensImage_unperturbed_noisy, kwargs_init, kwargs_fixed, kwargs_prior=kwargs_prior)


