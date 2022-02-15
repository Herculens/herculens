import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

from GRF_perturbations.Modules.Utils import gradient_descent,Spectrum_radial_averaging,jax_map
from GRF_perturbations.Modules.GRF_inhomogeneities_class import GRF_inhomogeneities_class
from GRF_perturbations.Modules.Surface_Brightness_class import Surface_brightness_class

class Inference_class:

    def __init__(self, GRF_inhomogeneities: GRF_inhomogeneities_class, Surface_brightness: Surface_brightness_class,
                 Grad_descent_max_iter=None, Grad_descent_learning_rate=None):
        """
        Initialize the class that handles inference of unperturbed source-lens model
        and inference of GRF parameters likelihood by surface brightness anomalies power spectrum.
        Parameters
        ----------
        GRF_inhomogeneities: GRF_inhomogeneities_class
            Class that handles lensing-related quantities of galaxy satellites
        Surface_brightness: Surface_brightness_class
            Class that generates mock surface brightness of the gravitational lens
        Grad_descent_max_iter: int
            Number of gradient descent iterations used for fitting unperturbed source-lens model
        Grad_descent_learning_rate: float
            Learning rate of gradient descent iterations used for fitting unperturbed source-lens model
        """

        self.GRF_inhomogeneities=GRF_inhomogeneities
        self.Surface_brightness=Surface_brightness
        self.max_iter=Grad_descent_max_iter
        self.lr_rate=Grad_descent_learning_rate

        if Grad_descent_max_iter is None:
            self.max_iter=1000

        if Grad_descent_learning_rate is None:
            self.lr_rate=5e-4

        # Function that simulates unperturbed mock for given arguments
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        self.simulate_unperturbed_image_pure= lambda model_kwargs: simulate_unperturbed_image(model_kwargs,Noise_flag=False)

        # Functionn that simulates perturbed mock for given GRF potential and predefined Source-lens unperturbed model
        self.simulate_perturbed_image=self.Surface_brightness.perturbed_image_getter

        # Source-lens parameters. Define kwargs<->args transformations, initial values and priors
        self.SL_parameters=self.Surface_brightness.parameters()

        #Gradient of loss used in grad descent
        self.image_loss_gradient = jax.grad(self.Loss_unperturbed_model)


    @partial(jax.jit, static_argnums=(0,))
    def Loss_unperturbed_model(self,args,data):
        """
        Chi^2 loss for given unperturbed Source-Lens arguments
        and 'data' gravitational lens surface brightness that they are to describe
        Parameters
        ----------
        args: jnp.ndarray real
            Arguments of Source light, Lens mass, Lens light assuming abscence of galaxy satellites
        data: jnp.ndarray real (self.Surface_brightness.pixel_number,self.Surface_brightness.pixel_number)
            The surface brightness of gravitation lens for which the parameters of source and lens 'args' are optimized
        Returns
        -------
            Chi^2: float
                negative log-likelihood for hypotheses that the gravitational lens surface brightness 'data'
                 was generated from the source and lens described by the parameters 'args'
        """

        kwargs=self.SL_parameters.args2kwargs(args)
        model=self.simulate_unperturbed_image_pure(kwargs)
        # Chi^2 loss
        return jnp.mean((data-model)**2/self.Surface_brightness.noise_var)

    @partial(jax.jit, static_argnums=(0,))
    def differentiable_fit_Surface_Brightness(self,data):
        """

        Parameters
        ----------
        data jnp.ndarray real (self.Surface_brightness.pixel_number,self.Surface_brightness.pixel_number)
            The surface brightness of gravitation lens for which the parameters of source and lens 'args' are optimized
        Returns
        -------
        args_max_likelihood: jnp.ndarray real
            Arguments of Source light, Lens mass, Lens light that result in Surface Brightness best describing the 'data'
        """

        # Compile pure function grad(Loss) for arguments of Source-Lens model
        model_loss_grad= jax.jit(lambda args: self.image_loss_gradient(args,data))
        # Initialise the parameters used to start gradient descent (important that it is np.array, not list)
        args_guess=self.SL_parameters.kwargs2args(self.Surface_brightness.kwargs_unperturbed_model)

        # Differentiable version of gradient descent algorithm
        args_max_likelihood=gradient_descent(model_loss_grad,args_guess,self.max_iter,self.lr_rate)

        return args_max_likelihood

    @partial(jax.jit, static_argnums=(0,))
    def compute_radial_spectrum(self,SB_anomalies):
        """
        Parameters
        ----------
        SB_anomalies:  jnp.ndarray real (self.Surface_brightness.pixel_number,self.Surface_brightness.pixel_number)
            Surface brightness anomalies for which you want to compute Radial Power spectrum
        Returns
        -------
        Radial_power_spectrum: jnp.ndarray real (len(frequencies))
            Array powers that describe power spectrum assuming isotropy of Surface brightness anomalies
        """

        # Leave only region with Einstein ring
        masked_anomalies = SB_anomalies * self.Surface_brightness.annulus_mask

        independent_spectrum_index = self.Surface_brightness.pixel_number//2

        # Since SB_anomalies are real, one half of Fourier image is a conjugate of the other.
        # We need only independent half of the Fourier image
        Fourier_image_half = jnp.fft.fft2(masked_anomalies)[:, :independent_spectrum_index]
        power_spectrum_half = jnp.abs(Fourier_image_half) ** 2
        # unitary normalisation of the Fourier transform (look np.fft norm 'ortho')
        normalized_spectrum_half = power_spectrum_half / self.Surface_brightness.annulus_mask.sum()

        k_grid_half = self.GRF_inhomogeneities.k_grid[:, :independent_spectrum_index]
        Radial_spectrum = Spectrum_radial_averaging(normalized_spectrum_half, k_grid_half, self.Surface_brightness.frequencies)

        return Radial_spectrum

    # TODO: comments and unittests
    @partial(jax.jit, static_argnums=(0,3,))
    def Anomalies_Radial_Power_Spectrum(self,GRF_params,unit_Fourier_image,Noise_flag=True):
        """

        Parameters
        ----------
        GRF_params
        unit_Fourier_image
        Noise

        Returns
        -------

        """
        # Simulate gravitational potential inhomogeneities induce by galaxy satellites
        GRF_potential=self.GRF_inhomogeneities.potential(GRF_params,unit_Fourier_image)

        # Jax does not tolerate randomness, but we want noise to be different for different GRFs
        noise_seed = jnp.round(jnp.abs(GRF_params[0] * (GRF_params[1] + 1) * (unit_Fourier_image[0, 1].real * 1e+3 + 1) * 1e+5)).astype(int)

        # Mock Surface brightness with noise and GRF potential perturbations
        Perturbed_SB_image=self.simulate_perturbed_image(GRF_potential,Noise_flag=Noise_flag,noise_seed=noise_seed)

        # Fit perturbed image with unperturbed model to infer the Surface brightness anomalies
        args_Unperturbed_SB=self.differentiable_fit_Surface_Brightness(Perturbed_SB_image)

        # Surface brightness of the gravitational lens that doesn't contain GRF perturbations or noise
        Unperturbed_SB_image=self.simulate_unperturbed_image_pure(self.SL_parameters.args2kwargs(args_Unperturbed_SB))

        # Lensing anomalies induced by GRF potential inhomogeneities
        SB_Anomalies=Perturbed_SB_image-Unperturbed_SB_image
        Radial_Power_Spectrum=self.compute_radial_spectrum(SB_Anomalies)

        return Radial_Power_Spectrum


    # TODO: comments for the function, unittests for differentiability,
    #  figure  out how to not use std inside the function Cause otherwise we either won't have enough statistics
    # Or the func would be not differentiable. Approach step by step improving uncertainty?
    @partial(jax.jit, static_argnums=(0,2,3,4))
    def GRF_Power_Spectrum_Loss(self,GRF_params,GRF_seeds_number,Spectra_Loss_function,Noise=True):
        """

        Parameters
        ----------
        GRF_params
        GRF_seeds_number
        Spectra_Loss_function:
            It should be Chi^2 with fixed uncertainties, which are estimated separately and
            Loss is optimized step by step. Otherwise function is too costful to differentiate
            or the number of spectra used for uncertainty estimation is highly insufficient
        Noise

        Returns
        -------

        """

        # Set of random Fourier images representing different GRF field realisations
        unit_Fourier_images=self.GRF_inhomogeneities.tensor_unit_Fourier_images[:GRF_seeds_number]

        # Simulate Radial Power spectra for Surface Brightness Anomalies images generated for every GRF potential realisation
        getter_SB_Anomalies_spectra=jax.jit(lambda unit_Fourier_image: self.Anomalies_Radial_Power_Spectrum(GRF_params,unit_Fourier_image,Noise))
        SB_Anomalies_spectra=jax_map(getter_SB_Anomalies_spectra,unit_Fourier_images)

        Loss=Spectra_Loss_function(SB_Anomalies_spectra)
        return Loss
