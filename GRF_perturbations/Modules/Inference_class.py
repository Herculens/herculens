import jax
import numpy as np
import jax.numpy as jnp
import time

from GRF_perturbations.Modules.Utils import gradient_descent,Spectrum_radial_averaging
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

        # Function that simulates unperturbed model for given arguments
        simulate_unperturbed_image = self.Surface_brightness.unperturbed_image_getter
        self.simulate_unperturbed_image_pure= lambda model_kwargs: simulate_unperturbed_image(model_kwargs,Noise_flag=False)
        # Source-lens parameters. Define kwargs<->args transformations, initial values and priors
        self.SL_parameters=self.Surface_brightness.parameters()

        #Gradient of loss used in grad descent
        self.image_loss_gradient = jax.grad(self.Loss_unperturbed_model)


    @jax.jit
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
        args_guess=np.array(self.SL_parameters.kwargs2args(self.Surface_brightness.kwargs_unperturbed_model))

        # Differentiable version of gradient descent algorithm
        args_max_likelihood=gradient_descent(model_loss_grad,args_guess,self.max_iter,self.lr_rate)

        return args_max_likelihood

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
