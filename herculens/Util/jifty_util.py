# This submodule contains utility functions to set up field models and 
# run variational inference strategies implemented in NIFTy.
# 
# Copyright (c) 2024, herculens and nifty developers and contributors

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

def jft_imports():
    try:
        import nifty.re as jft
    except ImportError:
        try:
            import nifty8.re as jft
        except ImportError:
            raise ImportError("The package `nifty` with the `.re` extension (version >=9.1.0) "
                              "- alternatively the older package nifty8 (version >=8.5.7) -"
                              "must be installed to use the CorrelatedField class. "
                              "See https://github.com/NIFTy-PPL/NIFTy to install it.")
    try:
        from nifty.re.tree_math import ShapeWithDtype
    except ImportError:
        try:
            from nifty8.re.tree_math import ShapeWithDtype
        except ImportError:
            pass  # no need to treat exceptions as we did it above already
    return jft, ShapeWithDtype


jft, ShapeWithDtype = jft_imports()


def prepare_correlated_field(
        key, num_pix_xy, border_xy, 
        kwargs_amplitude, 
        kwargs_fluctuations,
        kernel_type='powerlaw',
        correlation_type='power',
        matern_renorm_amp=False,
        num_pix_wl=1, 
        border_wl=0, 
        kwargs_fluctuations_wl=None,
        field_key='field',
        param_key_xy='xy_dim', 
        param_key_wl='wl_dim',
        correlation_type_wl='none',
        non_linearity='exp'
    ):
    """Utility that sets up a nifty.re correlated field forward model.

    Parameters
    ----------
    key : str
        Suffix (like a parameter name) added to the internal field parameter names
    num_pix_xy : int
        Number of pixels along each spatial dimensions (i.e. we assume a square grid).
    border_xy : int
        Number of pixels to crop on each sides of the field before it's returned by the model.
    kwargs_amplitude : dict
        Parameters of the field (see nifty.re documentation)
    kwargs_fluctuations : _type_
        Parameters of the field (see nifty.re documentation)
    kernel_type : str, optional
        Type of kernel to use for the field, either 'powerlaw' or 'matern', by default 'powerlaw'.
        If 'matern', the kwargs_fluctuations are given to the CorrelatedFieldMaker's
        method `add_fluctuations_matern()`, instead of `add_fluctuations()`.
        See the nifty.re documentation for more details. Note that the 'matern' kernel
        is only supported for the spatial dimensions, not for the spectral dimensions.
    num_pix_wl : int, optional
        Same as num_pix_xy but for the spectral (i.e. along wavelengths) dimensions, by default 1.
    border_wl : int, optional
        Same as border_xy but for the spectral (i.e. along wavelengths) dimension, by default 1.
    kwargs_fluctuations_wl : _type_, optional
        Same as kwargs_fluctuations for the spectral (i.e. along wavelengths) dimension
    non_linearity : str, optional
        Whether or not to apply a non-linearity to get specific behaviors, by default 'exp'.
        For instance, exponential non-linearity is useful to ensure non-negative values.

    Returns
    -------
    tuple of (jft.CorrelatedFieldMaker, jft.Model, int, int)
        Instance of the correlated field maker, the forward field model, 
        the total number of pixels in the spatial dimensions, 
        and the total number of pixels in the spectral dimensions.

    Raises
    ------
    ValueError
        If the non-linearity is not implemented.
    """
    if non_linearity not in ['exp', 'none']:
        raise ValueError(f"Non-linearity '{non_linearity}' not implemented.")
    if kernel_type not in ['powerlaw', 'matern']:
        raise ValueError(f"Kernel type '{kernel_type}' not supported. Supported types are 'powerlaw' and 'matern'.")
    
    # Correlated field parameters
    cfm = jft.CorrelatedFieldMaker(key + f'_{field_key}_')
    cfm.set_amplitude_total_offset(**kwargs_amplitude)

    # Setup the power-spectrum in the spectral dimension (if multi-band fitting)
    if num_pix_wl > 1:
        num_pix_wl_tot = num_pix_wl + 2 * border_wl
        cfm.add_fluctuations(
            [num_pix_wl_tot],
            distances=1./num_pix_wl_tot,  # only makes a difference to get proper units for wavenumbers and certain types of covariance kernels
            **kwargs_fluctuations_wl,
            prefix=param_key_wl+'_',  # prefix key to e.g. distinguish between multiple fields along different dimensions
            non_parametric_kind=correlation_type_wl,
            harmonic_type='fourier',
        )
    else:
        num_pix_wl_tot = 1

    # Setup the power-spectrum in the spatial dimensions
    num_pix_xy_tot = num_pix_xy + 2 * border_xy
    if kernel_type == 'powerlaw':
        cfm.add_fluctuations(
            [num_pix_xy_tot, num_pix_xy_tot],
            distances=1./num_pix_xy_tot,  # only makes a difference to get proper units for wavenumbers and certain types of covariance kernels
            **kwargs_fluctuations,
            prefix=param_key_xy+'_',  # prefix key to e.g. distinguish between multiple fields along different dimensions
            non_parametric_kind=correlation_type,
            harmonic_type='fourier',
        )
    elif kernel_type == 'matern':
        cfm.add_fluctuations_matern(
            [num_pix_xy_tot, num_pix_xy_tot],
            distances=1./num_pix_xy_tot,  # only makes a difference to get proper units for wavenumbers and certain types of covariance kernels
            renormalize_amplitude=matern_renorm_amp,
            **kwargs_fluctuations,
            prefix=param_key_xy+'_',  # prefix key to e.g. distinguish between multiple fields along different dimensions
            non_parametric_kind=correlation_type,
            harmonic_type='fourier',
        )
    # NOTE: in the pure nifty version, the distances are also 1/num_pix in the SimpleCorrelatedField model

    # Finalize the correlated field
    field_transform = cfm.finalize()

    # Apply non-linear transformation to get the final light model
    if non_linearity == 'exp':
        field_transform = ExpCroppedFieldTransform(key, field_transform, border_xy, border_wl)
        
    return cfm, field_transform, num_pix_xy_tot, num_pix_wl_tot


def prepare_light_correlated_field(*args, **kwargs):
    # just for backwards compatibility
    return prepare_correlated_field(*args, **kwargs)


def concatenate_fields(*fields):
    def operator(x):
        op = {}
        for field in fields:
            op.update(field(x))
        return op
    domain = {}
    for field in fields:
        domain.update(field.domain)
    return jft.Model(
        lambda x: operator(x),
        domain=domain,
    )


def NormalTransform(m, s, key):
    """m: mean, s: std"""
    #print(f"Normal prior for '{key}' ({m}, {s})")
    if not hasattr(m, 'shape'):
        shape = ()
    else:
        shape = m.shape
    return jft.Model(
        lambda x: {key: x[key]*s + m},
        domain={key: ShapeWithDtype(shape, jnp.float64)},
    )

def NormalTransform_inverse_func(x, m, s):
    """m: mean, s: std, to go some normal distribution to standard normal distribution"""
    return (x - m) / s

def LognormalTransform(m, s, key):
    """m: mean, s: std"""
    #print(f"Log-normal prior for '{key}' ({m}, {s})")
    log_s = jnp.sqrt(jnp.log1p((s/m)**2))
    log_m = jnp.log(m) - log_s**2/2.
    if not hasattr(m, 'shape'):
        shape = ()
    else:
        shape = m.shape
    return jft.Model(
        lambda x: {key: jnp.exp(x[key]*log_s + log_m)},
        domain={key: ShapeWithDtype(shape, jnp.float64)},
    )

def UniformTransform(l, h, key):
    """l: low, h: high"""
    #print(f"Uniform prior for '{key}' [{l}, {h}]")
    if not hasattr(l, 'shape'):
        shape = ()
    else:
        shape = l.shape
    return jft.Model(
        lambda x: {key: (h - l) * jstats.norm.cdf(x[key]) + l},
        domain={key: ShapeWithDtype(shape, jnp.float64)},
    )

def UniformTransform_inverse_func(x, l, h):
    """m: mean, s: std, to go from uniform distribution to standard normal distribution"""
    return jstats.norm.ppf( (x - l) / (h - l) )

def ExpCroppedFieldTransform(key, cf, bxy, bwl):
    #print(f"Exponential Cropped correlated field prior for '{key}'")
    def crop(x):
        x_ = cf(x)
        if bwl == 0:
            if bxy > 0:
                x_ = x_[bxy:-bxy, bxy:-bxy]
        else:
            x_ = x_[bwl:-bwl, :, :]
            if bxy > 0:
                x_ = x_[:, bxy:-bxy, bxy:-bxy]
        return x_
    return jft.Model(
        lambda x: {key: jnp.exp(crop(x))},  # here the target key is different from domain keys below
        domain=cf.domain,
    )

def numpyro_to_nifty_prior(
        numpyro_model, 
        model_args=(), 
        model_kwargs={},
        field_components=None,
        verbose=False,
    ):
    """Utility to convert a numpyro probabilistic model into a NIFTy field-like prior model, 
    by iterating through the model trace and converting all sampled parameters 
    with supported distributions (currently Normal, Log-normal and Uniform) 
    into the corresponding NIFTy field transforms. Such a conversion is needed 
    as NIFTy's VI algrithms always assume that the (latent) model parameters are
    sampled from a standard normal distribution.

    A few notes:
    - Only sampled parameters (with type 'sample') are converted.
    - Observed sites are skipped.
    - Fixed parameters and point-like parameters may not be properly handled yet.
    - Parameters that depends on other parameters in the model may not be properly handled yet.
    
    Parameters
    ----------
    numpyro_model : function
        A numpyro probabilistic model object containing a `numpyro.sample` statements.
    model_args : tuple, optional
        Positional arguments to pass to the numpyro model. Default is empty tuple.
    model_kwargs : dict, optional
        Keyword arguments to pass to the numpyro model. Default is empty dict.
    field_components : dict, optional
        A dictionary mapping field component names to their corresponding NIFTy field components (e.g. a CorrelatedFieldMaker instance). 
        This is used to directly add the NIFTy field models to the list of fields if the model parameters match the latent parameters of the field components.
    verbose : bool, optional
        Whether to print detailed information about the conversion process, by default False.

    Returns
    -------
    NIFTy field transform or concatenated field transforms
        A single field transform if only one parameter is found, 
        or a concatenation of multiple field transforms if several parameters are found.
        Each transform corresponds to a sampled parameter from the model trace.

    Raises
    ------
    ValueError
        If site name does not match the name in site properties,
        or if no sampled parameters are found in the model trace.
    NotImplementedError
        If a distribution type in the model is not supported. 
        Currently supported types are Normal, LogNormal, and Uniform.
    """
    if field_components is None:
        field_components = {}
    def _check_if_field(site_name):
        for field_name, field_component in field_components.items():
            for field_latent_param in field_component.latent_parameter_props.keys():
                if site_name == field_latent_param:
                    return field_name, field_component
        return None, None

    from numpyro import handlers
    import numpyro.distributions as dist
    # get the model trace, i.e. a description of the parameter space and priors
    trace = handlers.trace(
        handlers.seed(numpyro_model, jax.random.PRNGKey(0))
    ).get_trace(*model_args, **model_kwargs)
    # get the 
    # iterate through the parameters and build the corresponding NIFTy field-like priors
    fields = []
    for site_name, site_props in trace.items():
        if site_name != site_props['name']:
            # this should never happen, but we check it just in case
            raise ValueError(f"Site name '{site_name}' does not match the name in site properties '{site_props['name']}'.")
                
        # we skip all observed sites and non-sampled parameters
        if site_props['type'] != 'sample' or site_props['is_observed'] is True:
            # NOTE: this may be problematic for models that have fixed parameters, 
            # or point-like (free but not sampled) parameters
            if verbose:
                print(f"Skipping site '{site_name}' of type '{site_props['type']}' and is_observed={site_props['is_observed']}.")
            continue

        # check if the site is a latent variable of a field model
        field_name, field_component = _check_if_field(site_name)
        if field_component is not None:
            if verbose:
                print(f"Site '{site_name}' is a latent parameter of the field model '{field_name}'. Adding the model directly to the list of fields.")
            fields.append(field_component.nifty_model)
            continue
        
        # otherwise set the transform from the standard normal distribution
        else:
            dist_fn = site_props['fn']

            # Supported transforms
            # - Normal
            if isinstance(dist_fn, dist.Normal):
                if verbose:
                    print(f"Adding Normal prior for '{site_name}' with mean {dist_fn.loc} and std {dist_fn.scale}")
                fields.append(NormalTransform(dist_fn.loc, dist_fn.scale, site_name))
            
            # - Log-normal            
            elif isinstance(dist_fn, dist.LogNormal):
                # the formulas correspond to those in numpyro.distributions.continuous.LogNormal
                m = np.exp(dist_fn.loc + dist_fn.scale**2 / 2)
                s = np.sqrt( (np.exp(dist_fn.scale**2) - 1) * np.exp(2 * dist_fn.loc + dist_fn.scale**2) )
                if verbose:
                    print(f"Adding Log-normal prior for '{site_name}' with mean {dist_fn.loc} and std {dist_fn.scale} "
                      f"(which corresponds to mean {m} and standard deviation {s} in the NIFTy transform)")
                fields.append(LognormalTransform(m, s, site_name))
            
            # - Uniform
            elif isinstance(dist_fn, dist.Uniform):
                if verbose:
                    print(f"Adding Uniform prior for '{site_name}' with low {dist_fn.low} and high {dist_fn.high}")
                fields.append(UniformTransform(dist_fn.low, dist_fn.high, site_name))

            # Approximated transforms
            # - Truncated normal           
            elif (isinstance(dist_fn, dist.TwoSidedTruncatedDistribution) and
                  isinstance(dist_fn.base_dist, dist.Normal)):
                if verbose:
                    print(f"Adding *Normal* instead of *TruncatedNormal* prior for '{site_name}' with mean {dist_fn.base_dist.loc} and std {dist_fn.base_dist.scale}")
                fields.append(NormalTransform(dist_fn.base_dist.loc, dist_fn.base_dist.scale, site_name))

            # Any other unsupported transforms
            else:
                raise NotImplementedError(f"Distribution type '{type(dist_fn)}' not supported. Supported types are Normal, LogNormal and Uniform.")
        
    # concatenate all the fields to get the final prior model
    if len(fields) == 0:
        raise ValueError("No sampled parameters found in the model trace. Cannot convert to NIFTy field.")
    elif len(fields) == 1:
        fields = fields[0]
    else:
        fields = concatenate_fields(*fields)

    return fields


def prepare_nifty_likelihood(
        data,
        lens_image,
        nifty_prior,
        params2kwargs_fn,
        likelihood_mask=None,
    ):
    """Utility to build a NIFTy likelihood instance, by building the signal response
    (i.e. the forward model in nifty wording) from the lens image model and the priors, 
    and by setting up the likelihood mask operator to mask the data and the model before computing the likelihood.
    
    Parameters
    ----------
    lens_image : LensImage
        A lens image object containing the forward lensing model to convert into a NIFTy signal response.
    nifty_prior : NIFTy field transform
        The NIFTy field transform acting as a prior on the latent space,
        e.g. as returned by the numpyro_to_nifty_prior() function.
    params2kwargs_fn : function
        A user-defined function that is takes the output of the nifty_prior and 
        converts it into a dictionary of keyword arguments to pass to the lens_image.model()
        method. This is typically what is defined in a Numpyro probabilistic model class,
        but with calls to nifty-based components like the CorrelatedField factored out.
        Note that if no nifty-based field is used, the function should not be modified. 

        Here is an example:
        ```
        # Original function from the user-defined numpyro model class
        # which calls the source_field.model() to generate the pixelated source model
        # prior to lens_image.model() call.

        def params2kwargs(params):
            kwargs_lens = [
                {
                    'theta_E': params['lens_theta_E'],
                    'e1': params['lens_e1'],
                    'e2': params['lens_e2'],
                    'center_x': 0., #params['lens_center_x'],
                    'center_y': 0., #params['lens_center_y'],
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0': 0.0, 'dec_0': 0.0
                }
            ]
            kwargs_source = [{'pixels': source_field.model(params)}]
            }
            return {
                'kwargs_lens': kwargs_lens,
                'kwargs_source': kwargs_source,
            }

            
        # Slightly adapted function suitable to define the NIFTy-based likelihood
        # which does *not* call the source_field.model() - since it is already
        # called within the nifty priors - but simply grabs the source pixels 
        # from the parameters dictionary

        def params2kwargs_adapted(params):
            kwargs_lens = [
                {
                    'theta_E': params['lens_theta_E'],
                    'e1': params['lens_e1'],
                    'e2': params['lens_e2'],
                    'center_x': 0., #params['lens_center_x'],
                    'center_y': 0., #params['lens_center_y'],
                },
                {
                    'gamma1': params['lens_gamma1'],
                    'gamma2': params['lens_gamma2'],
                    'ra_0': 0.0, 'dec_0': 0.0
                }
            ]
            # ONLY THIS LINE IS DIFFERENT in this particular example
            kwargs_source = [{'pixels': params['source_pixels']}]
            return {
                'kwargs_lens': kwargs_lens,
                'kwargs_source': kwargs_source,
            }
        # This params2kwargs_adapted() is the function to pass to numpyro_to_nifty_likelihood().
        ```
    data : array-like
        The observed data to fit, which is used to set up the likelihood.

    likelihood_mask : array-like, optional
        A boolean mask to apply to the data and the model before computing
        the likelihood, by default None (i.e. no mask is applied).
        This can be useful to exclude certain pixels from the likelihood computation, 
        for instance to exclude masked regions or to focus on specific areas of the image.

    Returns
    -------
    NIFTy likelihood model : 
         A NIFTy model that computes the negative log-likelihood of the data given the model parameters, which can be used in NIFTy's VI algorithms.
    """
    # Create a likelihood mask operator if any
    if likelihood_mask is None:
        apply_mask = lambda x: x 
        masked_data = data
    else:
        mask = likelihood_mask.astype(bool)
        apply_mask = lambda x: x[mask]
    
    # Apply the mask to the data
    masked_data = apply_mask(data)

    # Define a suitable function as expected by the NIFTy likelihood  
    def masked_model_and_inv_std_fn(x):
        model = lens_image.model(
            **params2kwargs_fn(nifty_prior(x))
        )
        masked_model = apply_mask(model)
        masked_inv_std = apply_mask(1. / jnp.sqrt(lens_image.C_D_model(model)))
        return (masked_model, masked_inv_std)
    
    # Wrap it as a nifty Model so that the domain gets properly tracked
    masked_model_and_inv_std =jft.Model(
        call=masked_model_and_inv_std_fn,
        domain=nifty_prior.domain,
    )

    # Defines the negative log-likelihood function
    neg_log_likelihood = jft.VariableCovarianceGaussian(masked_data).amend(masked_model_and_inv_std)

    return neg_log_likelihood


def run_vi(
        run_key,
        nifty_likelihood,
        n_total_iterations,
        algorithm='MGVI',  # 'MGVI'/'linear' or 'geoVI'/'nonlinear'
        n_samples=None,
        sample_mode=None,
        init_key=None,
        init_position=None,
        init_spread_factor=0.1,
        quick_run=False,
        output_dir=None,
        callback_fn=None,
        draw_linear_kwargs=None,
        nonlinearly_update_kwargs=None,
        kl_kwargs=None,
    ):
    """Function that sets up and run variational inference strategies implemented in NIFTy.
    The two families of algorithms currently supported are:
    - Metric Gaussian Variational Inference (MGVI, a.k.a. 'linear VI')
    - Geometric Variational Inference (geoVI, a.k.a. 'non-linear VI')
    (TODO: add references).

    These algorithms work particularly well for a lens model that contains 
    field components such as a CorrelatedField model for pixelated components.

    For complete details about the algorithms and specific settings, please see:
    - NIFTy documentation:
    - NIFTy source code, in particular the submodule `nifty.re.optimize_kl`

    Parameters
    ----------
    run_key : jax.random.PRNGKey
        Key to initialize VI sampling.
    nifty_likelihood : nifty.re.Likelihood
        _description_
    n_total_iterations : int
        Total number of VI iterations to run.
    algorithm : str, optional
        Type of VI algorithm to use, either 'MGVI'/'linear' or 'geoVI'/'nonlinear', by default 'MGVI'.
    n_samples : int or function, optional
        Number of samples to use at each iteration. Can be a fixed integer or a function that
    sample_mode : str or function, optional
        Whether to use resampling or not at each iteration. Can be a fixed string ('linear_resample', 'linear_sample', 'nonlinear_resample', 'nonlinear_sample') or a function that takes the iteration number as input and returns the corresponding string. If None (default), some nice settings are used that work well for lens modeling, which typically use resampling at the beginning of the process and then switch to sampling only.
    init_position : dict or jft.Vector, optional
        Initial position in the latent parameter space. If None, it is randomly drawn from the latent prior, by default None.
    init_spread_factor : float, optional
        Factor to multiply the initial position drawn from the latent prior, typically to reduce the initial spread if < 1, by default 0.1.
    quick_run : bool, optional
        Whether to use settings that are suitable for a quick run (e.g. for testing or debugging).
    output_dir : str, optional
        Directory to store NIFTy outputs such as stats and last VI samples, by default None (i.e. no output is stored).
    callback_fn : function, optional
        A callback function that is called at each iteration with the current samples and other info, which can be used to store intermediate results (e.g. plots) during the VI process, by default None (i.e. no intermediate results are stored).
    draw_linear_kwargs : dict, optional
        Keyword arguments to pass to the underlying linear sampling step.
        See NIFTy source code for more details about the supported settings.
        If None (default), some nice settings are used that work well for lens modeling.
    nonlinearly_update_kwargs : dict, optional
        Keyword arguments to pass to the underlying linear sampling step.
        See NIFTy source code for more details about the supported settings.
        If None (default), some nice settings are used that work well for lens modeling.
    kl_kwargs : dict, optional
        Keyword arguments to pass to the underlying KL optimization step.
        See NIFTy source code for more details about the supported settings.
        If None (default), some nice settings are used that work well for lens modeling.

    Returns
    -------
    tuple of (samples_vi, latent_samples_vi, final_state_vi)
        - samples_vi: the VI samples in the target parameter space (i.e. transformed from the latent space by nifty_prior)
        - latent_samples_vi: the VI samples in the latent parameter space (i.e. the output of the NIFTy optimization)
        - final_state_vi: the final state of the NIFTy optimization, which contains useful information about the optimization process and can be used for diagnostics and analysis.

    Raises
    ------
    ValueError
        If the algorithm type is not recognized. Supported types are 'MGVI'/'linear' and 'geoVI'/'nonlinear'.
    """
    # Get the initial position in parameter space (this could be a set of samples as well)
    if init_position is None:
        # drawing initial value from the latent prior (multivariate standard normal distribution)
        init_position = jft.random_like(init_key, nifty_likelihood.domain)

        # multiply by some factor, typically to reduce the initial spread if < 1
        for k in init_position.keys():
            init_position[k] *= init_spread_factor

        # wrapping the dictionary to a jifty Vector
        init_position = jft.Vector(init_position)

    elif not isinstance(init_position, jft.Vector):
        init_position = jft.Vector(init_position)

    # Set th number of samples
    # NOTE: 2 x the number of samples is effectively used and returned (it is called "mirror sampling")
    if n_samples is None:
        if quick_run is True:
            n_samples = 16
        elif n_samples is None:
            def n_samples(iiter):
                if iiter < n_total_iterations // 3:
                    return 4
                elif iiter < 2 * n_total_iterations // 3:
                    return 8
                elif iiter < n_total_iterations - 1:
                    return 16
                else:
                    return 64
            
    # Choose the type of VI algorithm
    if algorithm.lower() in ('geovi', 'nonlinear'):
        base_method = 'nonlinear'
    elif algorithm.lower() in ('mgvi', 'linear'):
        base_method = 'linear'
    else:
        raise ValueError(f"Algorithm type '{algorithm}' not recognized. Supported samplers are 'geoVI' and 'MGVI'.")

    # Choose if there is resampling (see NIFTy doc for more details)
    if quick_run is True:
        sample_mode = f'{base_method}_sample'
    elif sample_mode is None:
        def sample_mode(iiter):
            # just a suggestion from the NIFTy experts
            if iiter < 5:
                return f'{base_method}_resample'
            else:
                return f'{base_method}_sample'

    # Choose settings of each underlying sampling step
    # Below are some specific values that were found to work well for lens modeling.
    if draw_linear_kwargs is None:
        draw_linear_kwargs = dict(
            cg_name="Linear sampling",
            cg_kwargs=dict(
                absdelta=0.05, 
                maxiter=100 if not quick_run else 10
            ),
        )
    if nonlinearly_update_kwargs is None:  # only relevant for geoVI
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(
                name="Non-linear sampling", 
                cg_kwargs=dict(name=None),
                xtol=0.001,
                maxiter=15 if not quick_run else 3,
            )
        )
    if kl_kwargs is None:
        kl_kwargs = dict(
            minimize_kwargs=dict(
                name="KL minimization", 
                cg_kwargs=dict(name=None), 
                absdelta=0.5,
                maxiter=10 if not quick_run else 3,
            )
        )

    # Run the VI optimization
    # NOTE: most of the arguments can be fixed variable or functions (the input being the iteration number) such that settings can be dynamically varied during the process
    latent_samples_vi, final_state_vi = jft.optimize_kl(
        nifty_likelihood, 
        init_position, 
        key=run_key,
        n_total_iterations=n_total_iterations,
        n_samples=n_samples,
        draw_linear_kwargs=draw_linear_kwargs,
        nonlinearly_update_kwargs=nonlinearly_update_kwargs,
        kl_kwargs=kl_kwargs,
        sample_mode=sample_mode,
        odir=output_dir,  # this is where nifty will store some stats and last VI samples
        resume=False,

        # NOTE: to get intermediate results at each iteration, we need to specify a callback function that saves the samples and other info at each iteration
        callback=callback_fn,
    )

    # transform the latent parameter values to target parameter values
    samples_vi = jax.vmap(nifty_prior)(latent_samples_vi.samples)

    return samples_vi, latent_samples_vi, final_state_vi
