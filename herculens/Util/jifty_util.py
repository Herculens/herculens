import jax.numpy as jnp
import jax.scipy.stats as jstats

import nifty8.re as jft
from nifty8.re.tree_math import ShapeWithDtype


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

def prepare_light_correlated_field(key, light_model, border_xy, 
                                   kwargs_amplitude, kwargs_fluctuations,
                                   num_pix_wl=0, border_wl=0, kwargs_fluctuations_wl=None):
    # TODO: implement multi-band correlated field
    
    # retrieve the number of pixels from the LightModel instance
    num_pix_xy = light_model.pixel_grid_settings.get('num_pixels', None)
    if num_pix_xy is None:
        raise ValueError("Number of pixels should be set in the provided LightModel.")
    
    # Correlated field parameters
    cfm = jft.CorrelatedFieldMaker(key + '_field_')
    cfm.set_amplitude_total_offset(**kwargs_amplitude)

    # Setup the power-spectrum in the spectral dimension (if multi-band fitting)
    if num_pix_wl > 0:
        num_pix_wl_tot = num_pix_wl + 2 * border_wl
        cfm.add_fluctuations(
            [num_pix_wl_tot],
            distances=1./num_pix_wl_tot,  # only makes a difference to get proper units for wavenumbers and certain types of covariance kernels
            **kwargs_fluctuations_wl,
            prefix='wav_dim_',  # prefix key to e.g. distinguish between multiple fields along different dimensions
            non_parametric_kind='power',
        )

    # Setup the power-spectrum in the spatial dimensions
    num_pix_xy_tot = num_pix_xy + 2 * border_xy
    cfm.add_fluctuations(
        [num_pix_xy_tot, num_pix_xy_tot],
        distances=1./num_pix_xy_tot,  # only makes a difference to get proper units for wavenumbers and certain types of covariance kernels
        **kwargs_fluctuations,
        prefix='xy_dim_',  # prefix key to e.g. distinguish between multiple fields along different dimensions
        non_parametric_kind='power',
    )
    # NOTE: in the pure nifty version, the distances are also 1/num_pix in the SimpleCorrelatedField model

    # Finalize the correlated field
    field_transform = cfm.finalize()

    # Apply non-linear transformation to get the final light model
    field_transform = ExpCroppedFieldTransform(key, field_transform, border_xy, border_wl)
        
    return cfm, field_transform

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
