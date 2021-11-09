# This is implementation of function generating Gaussian Random Field
#In a jaxified way (with tracing amplitude and power slope of GRF)
#The base of the code was taken from the package https://github.com/steven-murray/powerbox

#Basic imports
import numpy as np

#JAX
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

#Given length of image size and pixel_scale
#Returns grid of wavenumbers and configuration space grid step
def get_k_grid_dx(npix, pix_scl):
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
        array (npix,npix) of wavenumbers realisations
  dx: float
        step of grid in configuration space

  """
  #THIS IS WRONG BOXLENGTH
  #DON'T FORGET TO CHANGE IT AND RECOMPUTE ALL THE GRIDS
  boxlength=2 * np.pi * npix * pix_scl
  #probably the correct one
  #boxlength=npix * pix_scl

  #Grid step for real-space
  dx = float(boxlength) / npix
  #wavenumbers along side
  k_vector=np.fft.fftshift(np.fft.fftfreq(npix, d=dx)) * 2 * np.pi
  #wavenumbers 2d grid
  k_grid=np.sqrt(np.sum(np.meshgrid(*([k_vector ** 2] * 2)), axis=0))
  return k_grid,dx


def get_phase_realisation(npix,seed=None):
  "A random array which has Gaussian magnitudes and Hermitian symmetry"
  key=jax.random.PRNGKey(seed)
  #uneven number
  n=npix + 1 if (npix%2==0) else npix
  #2d grid
  size=[n]*2

  magnitude=jax.random.normal(key, shape=size)
  phase=2 * jnp.pi * jax.random.uniform(key,shape=size)

  #Make hermitian (why?)
  magnitude=(magnitude+magnitude[::-1,::-1])/jnp.sqrt(2)
  phase=(phase-phase[::-1,::-1])/2 + jnp.pi

  phase_realisation=magnitude * (jnp.cos(phase) + 1j * jnp.sin(phase))

  if (npix%2==0):
    #why?
    phase_realisation=phase_realisation[:-1,:-1]

  return phase_realisation


def _adjust_phase(ft, npix, pix_scl):
    '''IDK what is that, some border conditions I guess,
        But it is needed to correct Configuration space image
        Otherwise, its just a lattice structure instead of continuous GRF'''
    #Fourier parameters
    dx = 2 * np.pi * pix_scl
    k_vector=np.fft.fftshift(np.fft.fftfreq(npix, d=dx)) * 2 * np.pi


    left_edge=k_vector[0]
    freq=np.fft.fftshift(np.fft.fftfreq(npix,d=2*np.pi/dx/npix))*2*np.pi
    #for scalar left-edge and 1row freq
    xp = np.array([np.exp( 1j * freq * left_edge)])
    #phase correction
    ft = ft * xp.T
    ft = ft * xp
    return ft


def get_jaxified_GRF(params,seed,npix,pix_scl):
  """
  The very jaxified GRF.
  Parameters
  ----------
  params: (float,float)
        These are [log(Amp),Power_slope] of desired GRF's spectrum Amp*k^(-Power_slope)
  seed: int (same as in numpy)
        The seed is used to generate GRF's phase realisation in Fourier space
  npix: int
        Number of pixels along side of the image
  pix_scl: float
        arcsec/pixel or any 'physical_unit/pixel'

  Returns
  -------
  GRF: DeviceArray with shape (npix,npix)

  Examples
  -------
  >>> GRF=get_jaxified_GRF([0.,5.],1,100,0.08)
  >>> GRF.shape
  (100,100)
  >>> GRF_std=(lambda parameters: jnp.std(get_jaxified_GRF(parameters,1,100,0.08)))
  >>> jax.grad(GRF_std)([1.,5.])
  [DeviceArray(220.31397733, dtype=float64),DeviceArray(183.54880537, dtype=float64)]
  """

  logA,beta=params
  A=jnp.power(10.,logA)

  k_grid,dx=get_k_grid_dx(npix,pix_scl)
  #we will do ps=k^(-beta) with gradient ps_grad=-ln(k)*k^(-beta),
  #we want both to be 0 in center where k=0
  #So we set k=1 to avoid divergence with setting k=0 or k=inf
  k_grid[npix//2,npix//2]=1
  #In the same time we mask out the place where k=0, to exclude it from equation
  mask_center=np.ones_like(k_grid)
  mask_center[npix//2,npix//2]=0


  #workaround: sqrt(Power spectrum) in one operation
  #Jax gets nans when we do operations separately like sqrt(power*mask)
  sqrt_power_array=jnp.sqrt(A)*jnp.power(k_grid,-beta/2.)*mask_center
  #Random phase realisation in Fourier space
  phase_realisation=get_phase_realisation(npix,seed)

  #Create fourier image with random phases
  Fourier_image=sqrt_power_array*phase_realisation
  #Fourier_image=power_array*phase_realisation

  #Go to configuration space
  Config_image=(npix**2)*jnp.fft.ifftshift(jnp.fft.ifftn(Fourier_image))

  #Adjust phases of the image
  Config_image=_adjust_phase(Config_image,npix, pix_scl)

  #Get rid of complexity
  return jnp.real(Config_image)
