import jax
from functools import partial

def purify_function(function,*static_args):
    '''

    Parameters
    ----------
    function: func(traced_argument,*static_args)
    static_args: whatever args the function may require

    Returns
    -------
    pure_function: pure_function(traced_argument)=func(traced_argument,*static_args)

    Explanation
    -------
    This is a workaround of a problem that Jax.partial(jax.jit,static_args=(...)) doesn't allow
    to use non-hashable static_args like, for example, numpy arrays

    The workaround is that we precompile function with all the static arguments to a pure state,
    So static arguments are stored inside the compiled function and the function becomes pure and differentiable

    Examples
    ------
    >>>def func(x,y):
    >>>    return x*y

    >>>func_pure=purify_function(func,10)
    >>>func_pure(2)
    20
    '''
    def pure_function(traced_arg):
      return function(traced_arg,*static_args)
    return jax.jit(pure_function)

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


def gradient_descent(gradient_function,initial_guess,max_iter,learning_rate):
  '''

  Parameters
  ----------
  gradient_function: func
            gradient of function to be minimised (X.shape)->(X.shape)
  initial_guess: number or np/jnp array
        argument X broadcastable in a way (X+X)->(2*X)
  max_iter: int
        Maximal number of iterations
  learning_rate: float
        Learning rate for X=X-learning_rate*gradient_function(X)
  Returns
  -------

  '''

  step_function= lambda _,X: X-learning_rate*gradient_function(X)

  return jax.lax.fori_loop(0,max_iter,step_function,initial_guess)
