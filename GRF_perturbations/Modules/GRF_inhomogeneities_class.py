#Basic imports
import numpy as np
from copy import deepcopy
import math

#JAX
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

# Its very strange because Box-Muller function passes normality test, but some of  Fourier images pixels don't
def Box_Muller_transform():
    # Polar Box-Muller transform samples independent standard normal deviates
    #Samples real and imaginary parts of Fourier image for unit power
    s = 1.1
    while s > 1. :
        u = np.random.uniform (-1.,1.)
        v = np.random.uniform (-1.,1.)
        s = u**2. + v**2.
    fac = np.sqrt(-2.*np.log(s)/s)
    z1 = u*fac
    z2 = v*fac
    return z1,z2

class GRF_inhomogeneities_class:
    """A class that handles generation of GRF gravitational potential inhomogeneities that
    correspond to potential of galaxy satellites. Also generates supplementary quantities
    like differential deflection and convergence fields and field variances from Parseval's theorem"""

    def __init__(self,pixel_number: int,pixel_scale: float,Phase_seeds_number: int):
        """
        Initialize model for given grid pixelisation, resolution and number GRF random seeds
        Parameters
        ----------
        pixel_number: int (even)
            Lens plane grid is a square with side of 'pixel_number' pixels
        pixel_scale: float
            Resolution 'arcsec/pixel'
        Phase_seeds_number: int
            Number of random seeds for generation of different GRFs. Needed to precompute
            Fourier images for unit power spectrum.
        """
        
        # Even shaped square grid facilitates geometrical trnasformations, but should be generalised in the future
        assert (type(pixel_number) == int) and (pixel_number % 2 == 0)

        self.pixel_number=pixel_number
        self.pixel_scale=pixel_scale
        self.Phase_seeds_number=Phase_seeds_number

        #1d Spatial frequencies
        k_vector = np.fft.fftfreq(pixel_number, pixel_scale)
        #2d matrices for wavevector components
        kx, ky = np.meshgrid(k_vector, k_vector)
        #wavevector amplitudes
        self.k_grid = np.sqrt(kx ** 2 + ky ** 2)
        #Soften zero wavenumber, so spectrum doesn't diverge
        self.nonsingular_k_grid=deepcopy(self.k_grid)
        self.nonsingular_k_grid[0, 0] = 1

        #Precompute Fourier images for unit spectrum, because jax doesn't tolerate while loops inside pure functions
        self.tensor_unit_Fourier_images = np.zeros((self.Phase_seeds_number, self.pixel_number, self.pixel_number),dtype=complex)
        for i in range(self.Phase_seeds_number):
            self.tensor_unit_Fourier_images[i] = self.sample_unit_Fourier_image(random_seed=i)

        # Return seed to the fixed one
        np.random.seed(42)

    # TODO: Change sampling with np.random.seed to rng = np.random.RandomState(2021)
    #  (https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f)
    def sample_unit_Fourier_image(self,random_seed):
        """
        Samples random Fourier image for unit power spectrum of GRF inhomogeneities
        Parameters
        ----------
        random_seed: int
        seed for random Fourier phase generation

        Returns
        -------
        unit_Fourier_image: np.ndarray complex (pixel_number,pixel_number)
        Fourier image for unit power spectrum sampled using Polar Box-Muller transform
        """
        np.random.seed(random_seed)

        Fourier_image = np.zeros([self.pixel_number, self.pixel_number], dtype='cfloat')  # Empty matrix to be filled in for the Fourier plane
        j = 0 + 1j  # Defining the complex number

        half_length=self.pixel_number//2

        #Define top part of Fourier image and obtain bottom as complex conjugate
        for y in range(half_length+1):
            for x in range(self.pixel_number):

                #Zero mean value of the field
                if (y==0) and (x==0):
                    Fourier_image[y,x]=0.
                    continue

                # Real and Imaginary parts
                z1, z2 = Box_Muller_transform()

                # Points where Fourier image should be real for configuration image to be real
                if (np.isin(y, [0, half_length]) and np.isin(x, [0, half_length])):
                    Fourier_image[y, x] = z1
                else:
                    Fourier_image[y, x] = (z1 + j * z2) / np.sqrt(2)

                #Complex conjugation
                Fourier_image[-y, -x] = Fourier_image[y, x].conjugate()

        return Fourier_image

    def nonsingular_Power_spectrum(self,Spectrum_parameters):
        """
        Generates 2d matrix of power spectrum A*k^(-beta) for given [log(A),beta]
        The spectrum is softened in k=0 to not diverge
        Parameters
        ----------
        Spectrum_parameters: [float,float]
            [log(A),Beta] as log(Amplitude) and minus Power slope of the power spectrum

        Returns
        -------
        Power_spectrum: jnp.ndarray real (pixel_number,pixel_number)
            Softened power spectrum for given A and Beta
        """
        logA,Beta=Spectrum_parameters

        Amplitude = jnp.power(10., logA)
        Power_spectrum = Amplitude * jnp.power(self.nonsingular_k_grid, -Beta)

        return Power_spectrum

    def field_variance(self,Spectrum_parameters,field='potential'):
        """
        Computes variance of a given field using Parseval's theorem
        The result is the theoretical variance, i.e. the field variance averaged over infinite random realisations
        Parameters
        ----------
        Spectrum_parameters: [float,float]
            [log(A),Beta] as log(Amplitude) and minus Power slope of the power spectrum

        field: str
            Variance of which field to compute. Fields: 'potential','alpha_y','alpha_x','kappa'

        Returns
        -------
        Variance: float
            Variance of a given field for given power spectrum parameters
        """

        # Power spectrum softened in wavevector=0
        potential_Power_spectrum = self.nonsingular_Power_spectrum(Spectrum_parameters)

        # Zero mean mask
        mask=np.ones_like(potential_Power_spectrum)
        mask[0,0]=0
        potential_Power_spectrum*=mask

        if field=='potential':
            return potential_Power_spectrum.sum()

        # Derivative in Config space is multiplication by k in Fourier space
        k_vector = np.fft.fftfreq(self.pixel_number, self.pixel_scale)
        kx, ky = np.meshgrid(k_vector, k_vector)

        if field=='alpha_y':
            derivative_factor = (2 * np.pi * ky)**2  # d(psi)/dy
        elif field=='alpha_x':
            derivative_factor = (2 * np.pi * kx) ** 2  # d(psi)/dy
        elif field=='kappa':
            derivative_factor = (1 / 4) * (2 * np.pi) ** 4 * (kx ** 4 + ky ** 4) # laplacian(psi)/2
        else:
            raise ValueError("field should be one of: 'potential','alpha_y','alpha_x','kappa'")

        return (potential_Power_spectrum*derivative_factor).sum()


    def potential(self,Spectrum_parameters,unit_Fourier_image):
        """
        Function generating gravitational potential of galaxy satellites
        in the model of GRF inhomogeneities.
        We have to pass Fourier image explicitly, to eventually compile the function
        to a pure one: potential(Spectrum_parameters)
        Parameters
        ----------
        Spectrum_parameters: [float,float]
            [log(A),Beta] as log(Amplitude) and minus Power slope of the power spectrum

        unit_Fourier_image: np.ndarray complex (pixel_number,pixel_number)
            complex matrix obtained from self.sample_unit_Fourier_image

        Returns
        -------
        potential: np.ndarray real (pixel_number,pixel_number)
            GRF Gravitational potential
        """

        #Power spectrum softened in wavevector=0
        Power_spectrum=self.nonsingular_Power_spectrum(Spectrum_parameters)

        #Fourier image of GRF to generate. unit_Fourier_image[0,0]=0, so spectrum softening doesn't play a role
        Fourier_image=jnp.sqrt(Power_spectrum)*unit_Fourier_image
        #Conf image that needs proper norm
        Configuration_image = jnp.fft.ifftshift(jnp.fft.ifftn(Fourier_image))

        # Normalisation for Parseval's theorem (Power_spectrum.sum()=potential.var())
        Normalisation_factor = self.pixel_number*self.pixel_number
        potential = Normalisation_factor * Configuration_image.real

        return potential


    def alpha(self,Spectrum_parameters,unit_Fourier_image,direction='y'):
        """
        Differential deflection of GRF potential inhomogeneities for given direction
        Parameters
        ----------
        Spectrum_parameters: [float,float]
            [log(A),Beta] as log(Amplitude) and minus Power slope of the power spectrum

        unit_Fourier_image: np.ndarray complex (pixel_number,pixel_number)
            complex matrix obtained from self.sample_unit_Fourier_image

        direction: 'y' or 'x'
            deflection is derivative d(psi)/dy or d(psi)/dx correspondingly

        Returns
        -------
        alpha: np.ndarray real (pixel_number,pixel_number)
            differential deflection of GRF inhomogeneities for given direction
        """

        # Power spectrum softened in wavevector=0
        potential_Power_spectrum = self.nonsingular_Power_spectrum(Spectrum_parameters)

        #Derivative in Config space is multiplication by k in Fourier space
        k_vector = np.fft.fftfreq(self.pixel_number, self.pixel_scale)
        kx, ky = np.meshgrid(k_vector, k_vector)

        if direction=='y':
            derivative_factor=(2*np.pi*ky)**2 #d(psi)/dy
        elif direction=='x':
            derivative_factor=(2*np.pi*kx)**2 #d(psi)/dx
        else:
            raise ValueError("Derivative direction should be 'x' or 'y'")

        #Power spectrum of differential deflection for a given direction
        alpha_Power_spectrum=potential_Power_spectrum*derivative_factor

        # Fourier image of GRF to generate
        Fourier_image = jnp.sqrt(alpha_Power_spectrum) * unit_Fourier_image
        # Conf image that needs proper norm
        Configuration_image = jnp.fft.ifftshift(jnp.fft.ifftn(Fourier_image))

        # Normalisation for Parseval's theorem (Power_spectrum.sum()=potential.var())
        Normalisation_factor = self.pixel_number * self.pixel_number
        alpha = Normalisation_factor * Configuration_image.real


        return alpha

    def kappa(self,Spectrum_parameters,unit_Fourier_image):
        """
        Differential convergence of GRF potential inhomogeneities
        Parameters
        ----------
        Spectrum_parameters: [float,float]
            [log(A),Beta] as log(Amplitude) and minus Power slope of the power spectrum

        unit_Fourier_image: np.ndarray complex (pixel_number,pixel_number)
            complex matrix obtain
        Returns
        -------
        kappa: np.ndarray real (pixel_number,pixel_number)
            differential convergence of GRF inhomogeneities
        """

        # Power spectrum softened in wavevector=0
        potential_Power_spectrum = self.nonsingular_Power_spectrum(Spectrum_parameters)

        # Derivative in Config space is multiplication by k in Fourier space
        k_vector = np.fft.fftfreq(self.pixel_number, self.pixel_scale)
        kx, ky = np.meshgrid(k_vector, k_vector)

        #kappa is laplacian(psi)/2
        derivative_factor=(1/4)*(2*np.pi)**4*(kx**4+ky**4)

        # Power spectrum of differential convergence
        kappa_Power_spectrum = potential_Power_spectrum * derivative_factor

        # Fourier image of GRF to generate
        Fourier_image = jnp.sqrt(kappa_Power_spectrum) * unit_Fourier_image
        # Conf image that needs proper norm
        Configuration_image = jnp.fft.ifftshift(jnp.fft.ifftn(Fourier_image))

        # Normalisation for Parseval's theorem (Power_spectrum.sum()=potential.var())
        Normalisation_factor = self.pixel_number * self.pixel_number
        kappa = Normalisation_factor * Configuration_image.real

        return kappa

