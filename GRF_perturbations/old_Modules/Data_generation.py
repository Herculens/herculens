# Basic imports
import numpy as np

#Herculens
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.LensModel.lens_model import LensModel
from herculens.LensImage.lens_image import LensImage
from herculens.Parameters.parameters import Parameters

from GRF_perturbations.Modules.GRF_generation import get_jaxified_GRF
from GRF_perturbations.Modules.Jax_Utils import purify_function
from GRF_perturbations.Modules.Image_processing import differentiable_fit_image,compute_radial_spectrum

#Default lensing setup
default_source_light_model_list = ['SERSIC_ELLIPSE']
default_lens_mass_model_list = ['SIE', 'SHEAR','PIXELATED']
default_lens_light_model_list = []

#These are just mean values of the priors. Some have minor offsets from zero for smooth fitting start
kwargs_init = {
    'kwargs_lens': [{'theta_E': 1.5,'e1': 1e-3,'e2': 1e-3,'center_x': 1e-3,'center_y': 1e-3},
                    {'gamma1': 1e-3, 'gamma2': 1e-3, 'ra_0': 0.0, 'dec_0': 0.0}],
    'kwargs_source': [{'amp': 5.0,'R_sersic': 2.5,'n_sersic': 2.,'center_x': 0.,'center_y': 0.,'e1': 1e-3,'e2': 1e-3}],
}

#Priors from this article Park et al. 2021 Table 1 (Partially)
kwargs_prior = {
    'kwargs_lens': [{'theta_E': ['uniform', 1., 2.], 'e1': ['gaussian', 0, 0.2], 'e2': ['gaussian', 0, 0.2], \
                     'center_x': ['gaussian', 0, 0.102], 'center_y': ['gaussian', 0, 0.102]}, {'gamma1': ['uniform', -0.5, 0.5], 'gamma2': ['uniform', -0.5, 0.5]}],

    'kwargs_source': [{'amp': ['uniform',5.0,20.0], 'R_sersic': ['uniform',1e-3,5.], 'n_sersic': ['uniform',1e-3,4.0],\
                       'center_x': ['uniform', -1.0, 1.0], 'center_y': ['uniform', -1.0, 1.0],'e1': ['gaussian', 0, 0.2], 'e2': ['gaussian', 0, 0.2]}],
}


class Observation_conditions_class:

    def __init__(self,pixel_number,pixel_scale,PSF_FWHM,SNR,exposure_time,supersampling_factor=None,\
                 source_light_model_list=None,kwargs_source_light=None,\
                 lens_mass_model_list=None,kwargs_lens_mass=None,\
                 lens_light_model_list=None,kwargs_lens_light=None,\
                 annulus_mask_borders=None):

        #Set up PixelGrid
        self.pixel_number=pixel_number
        self.pixel_scale=pixel_scale
        self.pixel_grid=self.update_PixelGrid(pixel_number,pixel_scale)

        if supersampling_factor is None:
            supersampling_factor=1
        self.kwargs_numerics = {'supersampling_factor': supersampling_factor}

        #Set up Source and unperturbed lens parameters
        if (source_light_model_list is None) or (kwargs_source_light is None):
            source_light_model_list=default_source_light_model_list
            kwargs_source_light = [{'amp': 10.0, 'R_sersic': 1.2, 'n_sersic': 1.5, 'center_x': 0.4, 'center_y': 0.15,'e1':0.07,'e2':-0.1}]
        self.source_light_model_list=source_light_model_list
        self.kwargs_source_light=kwargs_source_light

        if (lens_mass_model_list is None) or (kwargs_lens_mass is None):
            lens_mass_model_list=default_lens_mass_model_list
            x_coords,y_coords=self.grid_coordinates
            kwargs_lens_mass = [{'theta_E': 1.6, 'e1': 0.15, 'e2': -0.04, 'center_x': 0.0, 'center_y': 0.0},\
                    {'gamma1': -0.01, 'gamma2': 0.03, 'ra_0': 0.0, 'dec_0': 0.0},\
                    {'x_coords': x_coords, 'y_coords': y_coords, 'psi_grid': np.zeros_like(x_coords)}]
        self.lens_mass_model_list=lens_mass_model_list
        self.kwargs_lens_mass=kwargs_lens_mass

        if (lens_light_model_list is None) or (kwargs_lens_light is None):
            lens_light_model_list=default_lens_light_model_list
            kwargs_lens_light=[{}]
        self.lens_light_model_list=lens_light_model_list
        self.kwargs_lens_light=kwargs_lens_light

        #These are true values for unperturbed lens-source model. Obviously unknown to us
        self.kwargs_data = {'kwargs_lens': kwargs_lens_mass[:-1], 'kwargs_source': kwargs_source_light,'kwargs_lens_light':kwargs_lens_light}

        #Annulus mask and Fourier frequencies vector
        if annulus_mask_borders is None:
            annulus_mask_borders=[0.5,3]

        radius=np.hypot(*self.pixel_grid.pixel_coordinates)
        rmin,rmax=annulus_mask_borders
        #Mask covering Einstein ring
        self.annulus_mask=((radius >= rmin) & (radius <= rmax)).astype(bool)

        #There is no sense to consider Fourier space frequncies referring to sizes
        #That are bigger than the thickness of the masked region
        full_freq_vector=np.fft.fftshift(np.fft.fftfreq(self.pixel_number,self.pixel_scale))[self.pixel_number//2:]
        #Index from which to consider frequencies
        self.init_freq_index=np.where(full_freq_vector>1/(rmax-rmin))[0][0]
        self.frequencies=full_freq_vector[self.init_freq_index:]

        #Lens images for noiseless setup
        LensImage_perturbed_noiseless,LensImage_unperturbed_noiseless=get_LensImages(self.pixel_grid,PSF_FWHM,1e-10,None,\
                                                                                     self.lens_mass_model_list,self.source_light_model_list,self.lens_light_model_list,self.kwargs_numerics)
        self.LensImage_perturbed_noiseless=LensImage_perturbed_noiseless
        self.LensImage_unperturbed_noiseless=LensImage_unperturbed_noiseless

        #Estimate background noise and noise variance
        simulate_unperturbed_image=self.unperturbed_image_getter
        Image_unperturbed_noiseless=simulate_unperturbed_image(self.kwargs_data,Noise_flag=False)
        bkg_noise_sigma=Image_unperturbed_noiseless.max()/SNR
        #Noise variance map for both Poisson and Gaussian noise
        self.noise_var=np.abs(Image_unperturbed_noiseless)/exposure_time+bkg_noise_sigma**2

        #Lens images for noisy setup
        LensImage_perturbed_noisy,LensImage_unperturbed_noisy=get_LensImages(self.pixel_grid,PSF_FWHM,bkg_noise_sigma,exposure_time,\
                                                                                     self.lens_mass_model_list,self.source_light_model_list,self.lens_light_model_list,self.kwargs_numerics)
        self.LensImage_perturbed_noisy=LensImage_perturbed_noisy
        self.LensImage_unperturbed_noisy=LensImage_unperturbed_noisy

    def update_PixelGrid(self,pixel_number,pixel_scale):
        self.pixel_number=pixel_number
        self.pixel_scale=pixel_scale

        half_size = pixel_number * pixel_scale / 2
        ra_at_xy_0 = dec_at_xy_0 = -half_size + pixel_scale / 2
        transform_pix2angle = pixel_scale * np.eye(2)
        kwargs_pixel = {'nx': pixel_number, 'ny': pixel_number,
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': transform_pix2angle}
        pixel_grid = PixelGrid(**kwargs_pixel)

        return pixel_grid

    @property
    def grid_coordinates(self):
        xgrid, ygrid = self.pixel_grid.pixel_coordinates
        x_coords = xgrid[0, :]
        y_coords = ygrid[:, 0]
        return x_coords,y_coords

    @property
    def parameters(self):

        #Priors from this article Park et al. 2021 Table 1 (Partially)
        kwargs_prior = {
            'kwargs_lens': [{'theta_E': ['uniform', 1., 2.], 'e1': ['gaussian', 0, 0.2], 'e2': ['gaussian', 0, 0.2], \
                     'center_x': ['gaussian', 0, 0.102], 'center_y': ['gaussian', 0, 0.102]}, {'gamma1': ['uniform', -0.5, 0.5], 'gamma2': ['uniform', -0.5, 0.5]}],

            'kwargs_source': [{'amp': ['uniform',5.0,20.0], 'R_sersic': ['uniform',1e-3,5.], 'n_sersic': ['uniform',1e-3,4.0],\
                       'center_x': ['uniform', -1.0, 1.0], 'center_y': ['uniform', -1.0, 1.0],'e1': ['gaussian', 0, 0.2], 'e2': ['gaussian', 0, 0.2]}],
        }

        #These are just mean values of the priors. Some have minor offsets from zero for smooth fitting start
        kwargs_init = {
            'kwargs_lens': [{'theta_E': 1.5,'e1': 1e-3,'e2': 1e-3,'center_x': 1e-3,'center_y': 1e-3},
                    {'gamma1': 1e-3, 'gamma2': 1e-3, 'ra_0': 0.0, 'dec_0': 0.0}],
            'kwargs_source': [{'amp': 5.0,'R_sersic': 2.5,'n_sersic': 2.,'center_x': 0.,'center_y': 0.,'e1': 1e-3,'e2': 1e-3}],
        }

        kwargs_fixed = {
            'kwargs_lens': [{}, {'ra_0': 0., 'dec_0': 0.}],  # fix origin of the external shear profile
            'kwargs_source': [{}],  # fix all source parameters
            }

        return Parameters(self.LensImage_unperturbed_noiseless, kwargs_init, kwargs_fixed, kwargs_prior=kwargs_prior)

    @property
    def GRF_getter(self):
        def get_GRF(GRF_params,GRF_seed):
            """
            get GRF potential for given [log(Amp),Power_slope],seed
            Parameters
            ----------
            params: (float,float)
                These are [log(Amp),Power_slope] of desired GRF's spectrum Amp*k^(-Power_slope)
            seed: int (same as in numpy)
                The seed is used to generate GRF's phase realisation in Fourier space

            Returns
            -------
            GRF: DeviceArray with shape (npix,npix)

            Examples
            -------
            >>> GRF=get_GRF([0.,5.],1)
            >>> GRF.shape
            (100,100)
            >>> GRF_std=(lambda parameters: jnp.std(get_GRF(parameters,1)))
            >>> jax.grad(GRF_std)([1.,5.])
            [DeviceArray(220.31397733, dtype=float64),DeviceArray(183.54880537, dtype=float64)]
            """
            return get_jaxified_GRF(GRF_params,GRF_seed,self.pixel_number,self.pixel_scale)
        return get_GRF

    @property
    def unperturbed_image_getter(self):
        def simulate_unperturbed_image(model_kwargs,Noise_flag=True,noise_seed=42):
                '''
                Parameters
                ----------
                model_kwargs: dict
                    dict with {'kwargs_lens': ..., 'kwargs_source': ...,'kwargs_lens_light':...}
                Noise_flag: bool
                    Presence of noise
                noise_seed: unit32
                    seed for noise generation

                Returns
                -------
                image: jnp.ndarray
                    array of shape (pixel_number,pixel_number) which represents the observed image generated for model_kwargs
                '''

                if Noise_flag:
                    LensImage_unperturbed=self.LensImage_unperturbed_noisy
                else:
                    LensImage_unperturbed=self.LensImage_unperturbed_noiseless

                return LensImage_unperturbed.simulation(noise_seed=noise_seed,**model_kwargs)
        return simulate_unperturbed_image

    @property
    def perturbed_image_getter(self):
        def simulate_perturbed_image(GRF_potential,model_kwargs,Noise_flag=True,noise_seed=42):
                '''
                Parameters
                ----------
                GRF_potential: jnp.ndarray
                    Potential of GRF perturbations as array of shape (pixel_number,pixel_number)
                model_kwargs: dict
                    dict with {'kwargs_lens': ..., 'kwargs_source': ...,'kwargs_lens_light':...}
                Noise_flag: bool
                    Presence of noise
                noise_seed: unit32
                    seed for noise generation

                Returns
                -------
                image: jnp.ndarray
                    array of shape (pixel_number,pixel_number) which represents the observed image generated for model_kwargs
                '''

                kwargs_lens = model_kwargs['kwargs_lens']+[{'pixels': GRF_potential}]

                if Noise_flag:
                    LensImage_perturbed=self.LensImage_perturbed_noisy
                else:
                    LensImage_perturbed=self.LensImage_perturbed_noiseless

                return LensImage_perturbed.simulation(kwargs_lens=kwargs_lens,
                                            kwargs_source=model_kwargs['kwargs_source'],
                                            kwargs_lens_light=model_kwargs['kwargs_lens_light'],
                                            noise_seed=noise_seed)
        return simulate_perturbed_image


def get_LensImages(pixel_grid,PSF_FWHM,noise_sigma,exposure_time,lens_mass_model_list,source_light_model_list,lens_light_model_list,kwargs_numerics):
    ''' Get object of LensModel class '''
    PSF_class=PSF(**{'psf_type': 'GAUSSIAN', 'fwhm': PSF_FWHM})
    pixel_number,_=pixel_grid.num_pixel_axes
    Noise_class=Noise(pixel_number, pixel_number, **{'background_rms': noise_sigma, 'exposure_time': exposure_time})

    perturbed_lens_image = LensImage(pixel_grid, PSF_class, noise_class=Noise_class,
                                lens_model_class=LensModel(lens_mass_model_list),
                                source_model_class=LightModel(source_light_model_list),
                                lens_light_model_class=LightModel(lens_light_model_list),
                                kwargs_numerics=kwargs_numerics)

    unperturbed_lens_image = LensImage(pixel_grid, PSF_class, noise_class=Noise_class,
                                lens_model_class=LensModel(lens_mass_model_list[:-1]),
                                source_model_class=LightModel(source_light_model_list),
                                lens_light_model_class=LightModel(lens_light_model_list),
                                kwargs_numerics=kwargs_numerics)

    return perturbed_lens_image,unperturbed_lens_image

def generate_data(GRF_params,GRF_seed,Observation_conditions,fit=True,Noise_flag=True):

    #Specifics of Observation_conditions
    get_GRF=Observation_conditions.GRF_getter
    simulate_perturbed_image=Observation_conditions.perturbed_image_getter
    parameters=Observation_conditions.parameters
    simulate_unperturbed_image=Observation_conditions.unperturbed_image_getter
    simulate_unperturbed_image_pure=lambda kwargs: simulate_unperturbed_image(kwargs,Noise_flag=False)
    differentiable_fit_image_pure=purify_function(differentiable_fit_image,simulate_unperturbed_image_pure,\
                                                  parameters.kwargs2args(Observation_conditions.kwargs_data),Observation_conditions.noise_var,parameters,1000,1e-4)
    compute_radial_spectrum_pure=purify_function(compute_radial_spectrum,Observation_conditions.annulus_mask,Observation_conditions.init_freq_index)

    #this is the true perturbation of the lens potential
    GRF_potential=get_GRF(GRF_params,GRF_seed)

    #data_image will play the role of the observed lens, that we are going to study
    data_image=simulate_perturbed_image(GRF_potential,Observation_conditions.kwargs_data,Noise_flag=Noise_flag,noise_seed=42)
    #Fit the data_image with unperturbed lens-source model to get a guess of kwargs_data, since we don't know them initially
    #One might use sophisticated trust-krylov optimization but it was found that grad_descent converge to the same values anyway

    if fit:
        fit_image=differentiable_fit_image_pure(data_image)
    else:
        fit_image=simulate_unperturbed_image_pure(Observation_conditions.kwargs_data)

    data_resid_spectrum=compute_radial_spectrum_pure(data_image-fit_image)


    return GRF_potential,data_image,fit_image,data_resid_spectrum
