import copy
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import morphology

from jaxtronomy.Util import param_util


def mask_from_pixelated_source(lens_image, parameters):
    src_idx = lens_image.SourceModel.pixelated_index
    kwargs_param_mask = copy.deepcopy(parameters.current_values(as_kwargs=True))
    pixels = kwargs_param_mask['kwargs_source'][src_idx]['pixels']
    pixels = np.zeros_like(pixels)
    pixels[3:-3, 3:-3] = 1.  # so the source plane is ones except in a small margin all around
    kwargs_param_mask['kwargs_source'][src_idx]['pixels'] = jnp.array(pixels)
    model_mask = lens_image.source_surface_brightness(kwargs_param_mask['kwargs_source'],
                                                      kwargs_lens=kwargs_param_mask['kwargs_lens'],
                                                      unconvolved=True, de_lensed=False)
    model_mask = np.array(model_mask)
    model_mask[model_mask < 0.1] = 0.
    model_mask[model_mask >= 0.1] = 1.
    model_mask = morphology.binary_opening(model_mask, iterations=10)
    model_mask = morphology.binary_dilation(model_mask, iterations=3)
    return model_mask

def pixelated_region_from_sersic(kwargs_sersic, use_major_axis=False,
                                 min_width=1.0, min_height=1.0, scaling=1.0):
    # TODO: support arbitrary smooth source profile
    c_x = kwargs_sersic['center_x']
    c_y = kwargs_sersic['center_y']
    r_eff = kwargs_sersic['R_sersic']
    if 'e1' in kwargs_sersic:
        e1 = kwargs_sersic['e1']
        e2 = kwargs_sersic['e2']
    else:
        e1 = e2 = 0.
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    a = r_eff / np.sqrt(q)  # semi-major axis
    r = r_eff * np.sqrt(q)  # product average radius
    print(f"r_eff={r_eff:.2f}, r={r:.2f}, a={a:.2f}")
    diameter = 2*a if use_major_axis else 2*r
    width = diameter * np.abs(np.cos(phi)) * scaling
    height = diameter * np.abs(np.sin(phi)) * scaling
    width = max(min_width, width)
    height = max(min_height, height)
    # the following dict is consistent with arguments of PixelGrid.create_model_grid()
    kwargs_pixelated_grid = {
        'grid_center': [float(c_x), float(c_y)],
        'grid_shape': [float(width), float(height)],
    }
    return kwargs_pixelated_grid

def R_omega(z, t, q, nmax):
    """Angular dependency of the deflection angle in the EPL lens profile.

    The computation follows Eqs. (22)-(29) in Tessore & Metcalf (2015), where
    z = R * e^(i * phi) is a position vector in the lens plane,
    t = gamma - 1 is the logarithmic slope of the profile, and
    q is the axis ratio.

    This iterative implementation is necessary, since the usual hypergeometric
    function `hyp2f1` provided in `scipy.special` has not yet been implemented
    in an autodiff way in JAX.

    Note that the value returned includes an extra factor R multiplying Eq. (23)
    for omega(phi).

    """
    # Set the maximum number of iterations
    # nmax = 10
    
    # Compute constant factors
    f = (1. - q) / (1. + q)
    ei2phi = z / z.conjugate()
    # Set the first term of the series
    omega_i = z  # jnp.array(np.copy(z))  # Avoid overwriting z ?
    partial_sum = omega_i

    for i in range(1, nmax):
        # Iteration-dependent factor
        ratio = (2. * i - (2. - t)) / (2. * i + (2 - t))
        # Current Omega term proportional to the previous term
        omega_i = -f * ratio * ei2phi * omega_i
        # Update the partial sum
        partial_sum += omega_i
    return partial_sum
