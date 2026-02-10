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
    print(f"Normal prior for '{key}' ({m}, {s})")
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
    print(f"Log-normal prior for '{key}' ({m}, {s})")
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
    print(f"Uniform prior for '{key}' [{l}, {h}]")
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
    print(f"Exponential Cropped correlated field prior for '{key}'")
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
