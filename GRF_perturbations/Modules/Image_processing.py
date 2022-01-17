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

def Spectrum_radial_averaging(spectrum,k_grid_half,frequencies):
    shape_spectrum = k_grid_half.shape
    length_radial = len(frequencies)

    # Mask rings with increasing integer radius (in pixels)
    Ring_masks = np.zeros((length_radial, shape_spectrum[0], shape_spectrum[1]))
    for i in range(length_radial - 1):
        # Chose ring of given radius
        Ring_masks[i] = np.logical_and(k_grid_half >= frequencies[i], k_grid_half < frequencies[i + 1])

    # max negative freq 6.25
    dk = frequencies[1] - frequencies[0]
    Ring_masks[-1] = np.logical_and(k_grid_half >= frequencies[-1], k_grid_half < frequencies[-1] + dk)

    # Number of pixels in ring
    pix_in_bins = Ring_masks.sum(axis=(1, 2))
    # Sum of image values in a ring with radius i
    scan_func = lambda _, Ring_mask: (1, jnp.sum(spectrum * Ring_mask))
    # Differentiable loop over rings
    sum_in_bins = jax.lax.scan(scan_func, 0, Ring_masks)[1]

    radial_spectrum = sum_in_bins / pix_in_bins
    return radial_spectrum

def Radial_profile(image,image_shape):
    '''

    Parameters
    ----------
    image: jnp.ndarray
        Image of shape (size_y,size_x)
    image_shape: (size_y,size_x)
        Shape of the image. Number of pixels is assumed to be even!!!

    Returns
    -------
    Radial_profile: jnp.ndarray
        Traced array of size np.minimum(size_y,size_x)

    Explanation
    -------
    For a given image Radial_profile is mean value in a ring depending on the radius of the ring
    The function is supposed to be casted into a pure one: Radial_profile_pure(image)
    So the gradients of the function could be obtained at will
    Examples
    -------
    >>>image_shape=(10,10)
    >>>image=np.random.normal(size=(image_shape))
    >>>Radial_profile_pure=jax.jit(lambda image: Radial_profile(image,image_shape))
    >>>Radial_profile_pure(image)
    DeviceArray([ 0.22578851, -0.26061518, -0.13495504,  0.16055809,-0.28665176], dtype=float64)
    >>>jax.grad(lambda image: jnp.mean(Radial_profile_pure(image)))(image)
    DeviceArray([[0., 0., 0., 0.00714286, ..., 0.00714286, 0., 0., 0.]], dtype=float64)
    '''

    assert image_shape[0]%2==0
    assert image_shape[1]%2==0

    y,x=np.indices(image_shape)
    center=np.array(image_shape)/2-0.5

    R=np.sqrt((y-center[0])**2+(x-center[1])**2)
    R=R.astype(int)
    length_radial_profile=np.minimum(*image_shape)//2

    #Mask rings with increasing integer radius (in pixels)
    Ring_masks=np.zeros((length_radial_profile,image_shape[0],image_shape[1]))
    for i in range(length_radial_profile):
        #Chose ring of given radius
        Ring_masks[i]=np.logical_and(R>=i,R<i+1)

    #Number of pixels in ring
    pix_in_bins=Ring_masks.sum(axis=(1,2))

    #Sum of image values in a ring with radius i
    scan_func= lambda _,Ring_mask: (1,jnp.sum(image*Ring_mask))
    #Differentiable loop over rings
    sum_in_bins=jax.lax.scan(scan_func,0,Ring_masks)[1]

    radial_profile=sum_in_bins/pix_in_bins
    return radial_profile


def compute_radial_spectrum(image,annulus_mask,k_grid,frequencies):
    '''
    Parameters
    ----------
    image: jnp.ndarray
        array of shape (size,size)
    annulus_mask: np.ndarray
        mask of shape (size,size). Leaves only Einstein ring region
    init_freq_index: int
        indent of spectra from zero
        spectral frequencies k<1/(rmax-rmin) refer to scales bigger than the ring, so don't make sense

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
    #Mask should have the same shape as image
    #it should be square
    assert annulus_mask.shape[0]==annulus_mask.shape[1]

    #Leave only region with Einstein ring
    masked_image=image*annulus_mask

    independent_spectrum_index=annulus_mask.shape[1]//2

    spectrum = jnp.fft.fft2(masked_image)[:, :independent_spectrum_index]
    power_spectrum = jnp.abs(spectrum) ** 2
    #unitary normalisation of the Fourier transform (look np.fft norm 'ortho')
    normalized_spectrum=power_spectrum/annulus_mask.sum()

    k_grid_half=k_grid[:,:independent_spectrum_index]
    Radial_spectrum=Spectrum_radial_averaging(normalized_spectrum,k_grid_half,frequencies)

    return Radial_spectrum

def compute_radial_spectrum_old(image,annulus_mask,init_freq_index,frequencies):
    '''
    Parameters
    ----------
    image: jnp.ndarray
        array of shape (size,size)
    annulus_mask: np.ndarray
        mask of shape (size,size). Leaves only Einstein ring region
    init_freq_index: int
        indent of spectra from zero
        spectral frequencies k<1/(rmax-rmin) refer to scales bigger than the ring, so don't make sense

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
    #Mask should have the same shape as image
    #it should be square
    assert annulus_mask.shape[0]==annulus_mask.shape[1]

    #Leave only region with Einstein ring
    masked_image=image*annulus_mask

    #Compute and center spectrum
    spectrum=jnp.abs(jnp.fft.fft2(masked_image))**2
    #unitary normalisation of the Fourier transform (look np.fft norm 'ortho')
    normalized_spectrum=spectrum/annulus_mask.sum()
    shift=annulus_mask.shape[0]//2
    Centered_Spectrum=jnp.roll(normalized_spectrum,shift,axis=(0,1))

    Radially_avg_spectrum=Radial_profile(Centered_Spectrum,annulus_mask.shape)

    #Ignore frequencies corresponding to scales bigger than mask
    return Radially_avg_spectrum[init_freq_index:]



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

    model_loss_function_pure=jax.jit(lambda args: model_loss_function(args,data,simulate_unperturbed_image_pure,\
                                                                  noise_var,parameters))
    '''
    def loss(args):
        kwargs=parameters.args2kwargs(args)
        model=simulate_unperturbed_image_pure(kwargs)

        return jnp.mean((data-model)**2/noise_var)
    '''
    loss=jax.jit(model_loss_function_pure)
    grad_loss=jax.jit(jax.grad(loss))
    hess_loss=jax.jit(jax.jacfwd(jax.jit(jax.jacrev(loss))))

    if initial_values is None:
        initial_values=parameters.initial_values()

    res = scipy_minimize(loss, initial_values,jac=grad_loss,hess=hess_loss, method=method)

    return parameters.args2kwargs(res.x)
