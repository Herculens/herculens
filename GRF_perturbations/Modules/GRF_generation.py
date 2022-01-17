#Code by github/egorssed
# get_Fourier_phase partially done by Giorgos Vernardos

#Basic imports
import numpy as np
from copy import deepcopy

#JAX
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

#Given length of image size and pixel_scale (arcsec/pixel)
#Returns grid of radial frequencies
def get_k_grid(npix, pix_scl):
  """
  Parameters
  ----------
  npix: int
        Number of pixels along side of the image
  pix_scl: float
        arcsec/pixel or any 'physical_unit/pixel'

  Returns
  -------
  k_grid: (ndarray)
        grid (npix,npix) of radial wavenumbers
  """

  k_vector=np.fft.fftfreq(npix,pix_scl)
  kx,ky=np.meshgrid(k_vector,k_vector)
  k_grid=np.sqrt(kx**2+ky**2)
  nonsingular_k_grid=deepcopy(k_grid)
  nonsingular_k_grid[0,0]=1

  return k_grid,nonsingular_k_grid

def Box_Muller_transform():
    s = 1.1
    while s > 1. :
        u = np.random.uniform (-1.,1.)
        v = np.random.uniform (-1.,1.)
        s = u**2. + v**2.
    fac = np.sqrt(-2.*np.log(s)/s)
    z1 = u*fac
    z2 = v*fac
    return z1,z2

#Uniformly sampled complex phases
def get_Fourier_phase(npix,seed):
    """
    Parameters
    ----------
    npix: int
        Number of pixels along side of the image
    seed: int
        seed for random cos,sin generation

    Returns
    -------
    Fourier phase grid: (npix,npix) complex
        grid of Fourier phases obtained using Box-Muller transform Polar form (lookup wiki)
    """
    np.random.seed(seed)
    #rng=np.random.default_rng(seed)

    Fourier_phases = np.zeros ([npix, npix], dtype='cfloat') # Empty matrix to be filled in for the Fourier plane
    j= 0 + 1j # Defining the complex number

    for y in range(npix):
        for x in range(npix):

            # Filling in the grid
            if x==0 and y==0: # Subtract mean instead of modyfing Fourier image
                Fourier_phases[y,x] = 1.0
                continue

            #phi=np.random.uniform(0,2*np.pi)
            #z1=np.cos(phi)
            #z2=np.sin(phi)
            z1,z2=Box_Muller_transform()

            # three points that need to be real valued to get a real image after FFT:
            if x== 0 and y==npix/2:
                Fourier_phases[y,x] = z1
            elif x==npix/2 and y==0:
                Fourier_phases[y,x] = z1
            elif x==npix/2 and y==npix/2:
                Fourier_phases[y,x] = z1
            else :
                Fourier_phases[y,x] = (z1+j*z2)/np.sqrt(2)

            Fourier_phases[-y,-x] = Fourier_phases[y,x].conjugate()

        if y>npix/2.:
            break

    return Fourier_phases

#Power spectrum that doesn't diverge in k=0.
#Hence, GRF's mean should be subtracted
def nonsingular_Power_spectrum(GRF_params,nonsingular_k_grid):

    Amplitude=jnp.power(10.,GRF_params[0])
    Power_spectrum=Amplitude*jnp.power(nonsingular_k_grid,-GRF_params[1])

    return Power_spectrum

#Get GRF in jax-differentiable manner with respect to GRF_params
def get_jaxified_GRF(GRF_params,nonsingular_k_grid,Fourier_phase_grid):

  #On zero frequency the power spectrum has nondiverging value
  #Hence mean of GRF should be subtracted afterwards (it is needed for differentiability)
  PS=nonsingular_Power_spectrum(GRF_params,nonsingular_k_grid)

  Fourier_image=jnp.sqrt(PS)*Fourier_phase_grid
  Configuration_image=jnp.fft.ifftshift(jnp.fft.ifftn(Fourier_image))

  #Normalisation for Parseval's theorem
  #In numpy default FFT normalisation is 'backward'
  Normalisation_factor=nonsingular_k_grid.size

  Normalised_GRF=Normalisation_factor*Configuration_image.real

  Zero_mean_GRF=Normalised_GRF-Normalised_GRF.mean()

  return Zero_mean_GRF
