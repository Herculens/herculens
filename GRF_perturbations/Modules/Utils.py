import jax
import numpy as np
import jax.numpy as jnp
from functools import partial


def gradient_descent(gradient_function,initial_guess,max_iter,learning_rate):
    '''
    Differentiable realisation of gradient_descent method. It means that for negative log-likelihood Loss(args)=-log(p(args | data))
    this function will estimate max likelihood args_ML(data) in such a way that you will be able to take derivative d(args_ML(data))/d(data)
    to propagate derivatives from surface brightness anomalies to power spectrum of GRF potential inhomogeneities
    Parameters
    ----------
    gradient_function: func(args: jnp.array([float]))->jnp.array([float])
              grad(Loss) that is gradient of negative log-likelihood to be minimised
    initial_guess: np.ndarray real
          the 'args' from which the gradient descent starts
    max_iter: int
          Number of gradient descent iterations used for fitting
    learning_rate: float
          Learning rate of gradient descent iterations used for fitting unperturbed source-lens model
    Returns
    -------
    args_max_likelihood: jnp.ndarray real
          Arguments that result in maximum likelihood in terms of Loss used in 'gradient_function' grad(Loss)
    '''

    # Step of gradient descent
    step_function= lambda _,X: X-learning_rate*gradient_function(X)

    # Gradient descent is depth-limited recursion.
    # This method introduces recursion that you can differentiate
    return jax.lax.fori_loop(0,max_iter,step_function,initial_guess)

# TODO: change arguments and realisation to use Surface_Brightness_class
"""
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
"""

def Spectrum_radial_averaging(power_spectrum_half,k_grid_half,frequencies):
    """
    Assuming isotropy of spectrum average it over plane waves direction.
    The function has to be so complex since we want to take jaxified derivatives d(Radial_power_spectrum)/d(spectrum_half)
    Parameters
    ----------
    power_spectrum_half: jnp.ndarray real
        Power spectrum matrix. For real Configuration image one half of Fourier is conjugate of the other half.
        Here we require only the independent part of power spectrum, e.g. for image shape (2*N,2*N) the spectrum shape is (2*N,N)
    k_grid_half: np.ndarray real
        Matrix of wavevector amplitudes. We require the part that would correspond to independent power spectrum
    frequencies: np.ndarray real
        Array of wavenumbers outlining borders of rings in which the averaging is carried out
        For frequencies=[k1,k2,k3,...] the rings borders radii will be [k1,k2),[k2,k3),...
    Returns
    -------
        Radial_power_spectrum: jnp.ndarray real (len(frequencies))
        Array of powers corresponding to wavevector amplitudes outlined by 'frequencies'
    """

    # Initialize geometric properties
    shape_spectrum = k_grid_half.shape
    length_radial = len(frequencies)

    # We can not use indexing of jnp.ndarray so our only way is to use tensor multiplication
    # 'Ring_masks' is a tensor of mask coverings the rings, where the spectrum will be averaged.
    # First dimension corresponds to radial borders of a ring,
    # second and third dimensions define the mask that covers the ring
    Ring_masks = np.zeros((length_radial, shape_spectrum[0], shape_spectrum[1]))
    for i in range(length_radial - 1):
        # Make a mask with radial borders |wavevector| in [frequencies[i],frequencies[i + 1])
        Ring_masks[i] = np.logical_and(k_grid_half >= frequencies[i], k_grid_half < frequencies[i + 1])

    # Define the outer ring
    dk = frequencies[1] - frequencies[0]
    Ring_masks[-1] = np.logical_and(k_grid_half >= frequencies[-1], k_grid_half < frequencies[-1] + dk)

    # Number of pixels in a ring
    pix_in_bins = Ring_masks.sum(axis=(1, 2))
    # Total power in a ring described by Ring_mask
    Power_in_Ring = lambda _, Ring_mask: (1, jnp.sum(power_spectrum_half * Ring_mask))
    # Differentiable loop over rings
    sum_in_bins = jax.lax.scan(Power_in_Ring, 0, Ring_masks)[1]

    # Average power in a ring is sum(Power)/len(Power)
    Radial_power_spectrum = sum_in_bins / pix_in_bins
    return Radial_power_spectrum


def jax_map(f, xs):
    '''
    Parameters
    ----------
    f: function(x)
    xs: collection [x1,x2,x3]
    Returns
    -------
    [f(x1),f(x2),f(x3)]
    Explanation
    -------
    Differentiable version of mapping a function over an array.
    Can be used to map function over matrix,
    Mapping is carried out over the first dimension in that case
    Examples
    -------
    >>> get_GRF=lambda GRF_seed: get_jaxified_GRF_pure(GRF_params,GRF_seed)
    >>> GRFs=jax_map(get_GRF,GRF_seeds)
    >>> print('GRF_seeds.shape',GRF_seeds.shape)
    GRF_seeds.shape (10,)
    >>> print('Function output shape',get_jaxified_GRF_pure(GRF_params,GRF_seed).shape)
    Function output shape (100, 100)
    >>> print('Mapping output shape',GRFs.shape)
    Mapping output shape (10, 100, 100)
    '''
    #Function (carry,value)->(carry,f(value)), with no interest in carry
    scan_func = lambda _,x: (1,f(x))
    #Jaxified loop over an array
    ys=jax.lax.scan(scan_func,0,xs)[1]
    return ys


#map function(logA,Beta,GRF_seed) over grid of arrays of logA,Beta,GRF_seeds
def jax_map_over_grid(function,logA_array,Beta_array,GRF_seeds):
    '''

    Parameters
    ----------
    function: function(logA,Beta,GRF_seed)->x
    logA_array: np.array of logA values
    Beta_array: np.array of Beta values
    GRF_seeds: np.array of GRF_seed values

    Returns
    -------
    x_grid: array(len(logA_array),len(Beta_array),len(GRF_seeds),x.shape[0],x.shape[1],...)
    function mapped over all three arrays

    Explanation
    -------
    Differentiable version of mapping a function over an grid of GRF_perturbation parameters.

    Examples
    -------
    >>> function=lambda logA,Beta,GRF_seed: jnp.array([logA*Beta,GRF_seed])
    >>> logA_array=np.arange(3)
    >>> Beta_array=np.arange(2)
    >>> GRF_seeds=np.arange(1)
    >>> x_grid=jax_map_over_grid(function,logA_array,Beta_array,GRF_seeds)
    >>> x_grid.shape
    (3, 2, 1, 2)
    >>> x_grid
    DeviceArray([[[[0, 0]],[[1, 0]]],[[[1, 0]],[[2, 0]]],[[[2, 0]],[[3, 0]]]], dtype=int64)
    '''

    #map function(logA,Beta,seed) over GRF_seeds
    def loop_over_seeds(logA,Beta):
        #func_of_seed(seed)=function(logA,Beta,seed)
        func_of_seed=partial(function,logA,Beta)
        return jax_map(func_of_seed,GRF_seeds)

    #map function(logA,Beta,seed) over Beta_array and GRF_seeds
    def loop_over_Betas(logA):
        #func_of_Beta(Beta)=map(function(logA,Beta,GRF_seed),GRF_seeds))
        func_of_Beta=partial(loop_over_seeds,logA)
        return jax_map(func_of_Beta,Beta_array)

    return jax_map(loop_over_Betas,logA_array)
