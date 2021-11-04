#JAX
import jax
import jax.numpy as jnp

#Basic imports
import numpy as np
import math

#Jaxtronomy
from herculens.LensModel.lens_model import LensModel
from herculens.LensImage.lens_image import LensImage
from herculens.LightModel.light_model import LightModel
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise


#Stuff needed to implement functions in a pure way for jaxification
from GRF_perturbations.Modules.Jax_Utils import purify_function,gradient_descent

from scipy.optimize import minimize as scipy_minimize


#Parameters to make args<->kwargs transformations
#parameters=get_parameters()

def Radial_profile(image,image_shape):
  '''

  Parameters
  ----------
  image: jnp.ndarray
        Image of shape (size_y,size_x)
  image_shape: (size_y,size_x)
        Shape of the image. If the shape is apriori known

  Returns
  -------
  Radial_profile: jnp.ndarray
        Traced array of size np.minimum(size_y,size_x)

  Explanation
  -------
  For a given image Radial_profile represents the dependence of axially averaged Flux on radius
    The function is constructed in a way that for a given image shape it can be casted
    to a pure function: Radial_profile_pure(image)

  Examples
  -------
  >>>image_shape=(10,15)
  >>>image=np.random.normal(size=(image_shape))
  >>>Radial_profile_pure=pure_function(Radial_profile,image_shape)
  >>>Radial_profile_pure(image)
  DeviceArray([ 0.64495406, -0.47196613,  0.47290311, -0.20336265,-0.01885612], dtype=float64)
  '''

  size_y,size_x=image_shape
  #Needed to compute only below the smallest radius of the image
  min_size=np.minimum(size_y,size_x)

  #Centered coordinates
  x=np.arange(size_x)-size_x/2+0.5
  y=np.arange(size_y)-size_y/2+0.5
  X,Y=np.meshgrid(x,y)

  #Matrix of radii
  R=np.sqrt(np.power(X,2)+np.power(Y,2))
  radius_size=math.ceil(min_size/2)

  #We can not assign a value of traced array, but we can multiply it by something
  #For given radius we multiply an image by a mask. For loop can be avoided using a high dimensional tensor.
  R_mask_tensor=np.zeros((radius_size,size_y,size_x))
  for i in range(radius_size):
      #Chose ring of given radius
      condition=(R>=i) & (R<i+1)
      #Set the mask for this radius
      R_mask_tensor[i]=np.where(condition,1,0)

  #Number of pixels on given radius (to get mean from sum)
  pixel_counter=R_mask_tensor.sum(axis=(1,2))

  #Summary flux in ring with given radius
  #If we multiply image with shape (x,y) by tensor with shape (r,x,y)
  #and sum it up to receive an array with shape (r). It is a way to avoid for loop.
  sum_in_rings=(image*R_mask_tensor).sum(axis=(1,2))

  #Average flux in ring with given radius
  radial_profile=sum_in_rings/pixel_counter

  return radial_profile

def compute_radial_spectrum(image,mask,mask_spectral_cut_index):
  '''
  Parameters
  ----------
  image: jnp.ndarray
        array of shape (size,size)
  mask: np.ndarray
        mask of shape (size,size). Leaves only Einstein ring region
  mask_spectral_cut_index: int -> indent of spectra from zero
  (if we mask out a ring, than spectral frequencies refering to sizes larger than
  the r_max-r_min do not make sense)

  Returns
  -------
  power_spectrum: jnp.ndarray
        array with shape (size//2-mask_spectral_cut_index)
  Explanation
  -------
  For a given image computes Power spectrum. Then computes radial profile of the power spectrum,
  which is axially averaged spectrum.
  Returns radial profile of the power spectrum, with indent from low frequencies that corresponds to the mask_spectral_cut_index

  Examples
  -------
  >>> compute_radial_spectrum(resid0,Radial_profile,mask,0)
  DeviceArray([2.75495046e+00, 7.38967978e+00, 9.19915373e+00,1.17016538e+01, 1.23753751e+01, ...], dtype=float64)
  >>> compute_radial_spectrum_pure=purify_function(compute_radial_spectrum,Radial_profile,mask,4)
  >>> compute_radial_spectrum_pure(resid0)
  DeviceArray([1.23753751e+01, 1.02814510e+01, ...], dtype=float64)
  '''
  #Mask should be the same shape as image
  #it should be square
  assert mask.shape[0]==mask.shape[1]


  #Leave only region with Einstein ring
  masked_image=image*mask

  #Compute and center spectrum
  spectrum=jnp.abs(jnp.fft.fft2(masked_image))**2
  #unitary normalisation of the Fourier transform (look np.fft norm 'ortho')
  normalized_spectrum=spectrum/mask.sum()
  shift=mask.shape[0]//2
  Centered_Spectrum=jnp.roll(normalized_spectrum,shift,axis=(0,1))

  Radially_avg_spectrum=Radial_profile(Centered_Spectrum,mask.shape)

  #Ignore frequencies, that are irrelevant for corresponding mask
  return Radially_avg_spectrum[mask_spectral_cut_index:]



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

#
def model_loss_function(args,data,simulate_smooth_image_pure,noise_var,parameters):
    ''' The function that is used as loss when we fit model's kwargs. Actually, it's MSE for data and model images'''
    kwargs=parameters.args2kwargs(args)
    model=simulate_smooth_image_pure(kwargs)
    return jnp.mean((data-model)**2/noise_var)

def scipy_fit_image(data,simulate_unperturbed_image_pure,noise_var,parameters,method='BFGS',initial_values=None):
    '''
    Differentiable gradient descent-based function
    Parameters
    ----------
    data: jnp.ndarray
        image to be fitted
    lens_image: LensModel object
        class used to simulate the model image for the fit
    noise_var: jnp.ndarray
        map of noise variances needed for chi^2

    Returns
    -------
    kwargs
        lens-source kwargs that are the results of the fit
    '''

    def loss(args):
        kwargs=parameters.args2kwargs(args)
        model=simulate_unperturbed_image_pure(kwargs)

        return jnp.mean((data-model)**2/noise_var)

    loss=jax.jit(loss)
    grad_loss=jax.jit(jax.grad(loss))
    hess_loss=jax.jit(jax.jacfwd(jax.jit(jax.jacrev(loss))))

    if initial_values is None:
        initial_values=parameters.initial_values()

    res = scipy_minimize(loss, initial_values,jac=grad_loss,hess=hess_loss, method=method)

    return parameters.args2kwargs(res.x)

def differentiable_fit_image(simulated_image,simulate_smooth_image_pure,args_guess,noise_var,parameters,max_iter,learning_rate):
    '''
    Differentiable gradient descent-based function
    Parameters
    ----------
    simulated_image: jnp.ndarray
            array of shape (size,size) to be fitted with unperturbed source and lens embedded into simulate_smooth_image_pure
    parameters: Parameters object
            object needed to transform args<->kwargs
    simulate_smooth_image_pure: function(smooth_kwargs)
            Pure version of simulate_smooth_image function

    Explanation
    -------
    The GRF fitting pipeline requires to differentiate the fitting. It means given the function args=fit_func(image)
    we eventually want to know grad(args)=grad(fit_func)(image), i.e. how would our fit change if we give change the image.
    This is a fully differentible function that carries out the fit of lens image. So we can get grad(fit_func) using jax

    Returns
    -------
    image: jnp.ndarray
        array of shape (size,size) which represents the fit of a given simulated_image
    '''

    #Purify the model_loss_function for a given fitting setup
    model_loss_function_pure=purify_function(model_loss_function,simulated_image,simulate_smooth_image_pure,noise_var,parameters)
    model_loss_grad=jax.grad(model_loss_function_pure)

    #Gradiend descent is but a recursion. Here its depth-limited and differentiable version
    args_fit=gradient_descent(model_loss_grad,args_guess,max_iter,learning_rate)
    kwargs_fit=parameters.args2kwargs(args_fit)

    #Model the fit back
    fit_image=simulate_smooth_image_pure(kwargs_fit)
    return fit_image
