import copy
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import morphology
from scipy import sparse
import findiff

from herculens.LightModel.light_model import LightModel
from herculens.LensImage.lens_image import LensImage
from herculens.Util import param_util
from herculens.Util.jax_util import BicubicInterpolator as Interpolator


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


def mask_from_smooth_source(lens_image, parameters, threshold=0.1):
    kwargs_param = parameters.current_values(as_kwargs=True)
    source_model = lens_image.source_surface_brightness(kwargs_param['kwargs_source'], de_lensed=True, unconvolved=True)
    source_model = np.array(source_model)
    binary_source = source_model / source_model.max()
    binary_source[binary_source < threshold] = 0.
    binary_source[binary_source >= threshold] = 1.
    lens_image_pixel = LensImage(lens_image.Grid, lens_image.PSF, 
                                 noise_class=lens_image.Noise,
                                 lens_model_class=lens_image.LensModel,
                                 source_model_class=LightModel(['PIXELATED']),
                                 lens_light_model_class=lens_image.LensLightModel,
                                 kwargs_numerics=lens_image._kwargs_numerics)

    kwargs_param_mask = copy.deepcopy(kwargs_param)
    kwargs_param_mask['kwargs_source'] = [{'pixels': jnp.array(binary_source)}]
    model_mask = lens_image_pixel.source_surface_brightness(kwargs_param_mask['kwargs_source'], 
                                                      kwargs_lens=kwargs_param_mask['kwargs_lens'],
                                                      unconvolved=True, de_lensed=False)
    model_mask = np.array(model_mask)
    model_mask[model_mask < 0.1] = 0.
    model_mask[model_mask >= 0.1] = 1.
    #model_mask = morphology.binary_opening(model_mask, iterations=10)
    #model_mask = morphology.binary_dilation(model_mask, iterations=3)
    return model_mask, binary_source


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

def build_DsD_operator(smooth_lens_image, smooth_kwargs_params, hybrid_lens_image=None):
    """this functions build the full operator from Koopmans 2005"""
    # data grid
    # x_coords, y_coords = smooth_lens_image.Grid.pixel_axes
    x_grid, y_grid = smooth_lens_image.Grid.pixel_coordinates
    num_pix_x, num_pix_y = smooth_lens_image.Grid.num_pixel_axes
    pixel_width = smooth_lens_image.Grid.pixel_width

    # pixelated lens model grid (TODO: implement interpolation on potential grid)
    # x_coords_pot, y_coords_pot = hybrid_lens_image.Grid.model_pixel_axes('lens')
    # x_grid_pot, y_grid_pot = hybrid_lens_image.Grid.model_pixel_coordinates('lens')

    # numerics grid, for intermediate computation on a higher resolution grid
    x_grid_num, y_grid_num = smooth_lens_image.ImageNumerics.coordinates_evaluate
    shape_num = tuple([int(np.sqrt(x_grid_num.size))]*2)  # ASSUMES SQUARE GRID!
    x_grid_num = x_grid_num.reshape(shape_num) # convert to 2D array
    y_grid_num = y_grid_num.reshape(shape_num) # convert to 2D array
    x_coords_num, y_coords_num = x_grid_num[0, :], y_grid_num[:, 0]
    
    # get the pixelated source in source plane,
    # on the highest resolution grid possible (it will use )
    smooth_source = smooth_lens_image.SourceModel.surface_brightness(
        x_grid_num, y_grid_num, smooth_kwargs_params['kwargs_source'])
    smooth_source *= pixel_width**2  # proper units
    interp_source = Interpolator(y_coords_num, x_coords_num, smooth_source)

    # compute its derivatives *on source plane*
    grad_s_x_srcplane = interp_source(y_grid_num, x_grid_num, dy=1)
    grad_s_y_srcplane = interp_source(y_grid_num, x_grid_num, dx=1)
    grad_s_srcplane = np.sqrt(grad_s_x_srcplane**2+grad_s_y_srcplane**2)

    # setup the Interpolator to read on data pixels
    interp_grad_s_x = Interpolator(y_coords_num, x_coords_num, grad_s_x_srcplane)
    interp_grad_s_y = Interpolator(y_coords_num, x_coords_num, grad_s_y_srcplane)

    # use the lens equation to ray shoot the coordinates of the data grid
    x_src, y_src = smooth_lens_image.LensModel.ray_shooting(
        x_grid, y_grid, smooth_kwargs_params['kwargs_lens'])

    # evaluate the resulting arrays on that grid
    grad_s_x = interp_grad_s_x(y_src, x_src)
    grad_s_y = interp_grad_s_y(y_src, x_src)
    grad_s = np.sqrt(grad_s_x**2+grad_s_y**2)

    # put them into sparse diagonal matrices
    D_s_x = sparse.diags([grad_s_x.flatten()], [0])
    D_s_y = sparse.diags([grad_s_y.flatten()], [0])

    # compute the potential derivative operator as two matrices D_x, D_y
    step_size = pixel_width # step size
    order = 1 # first-order derivative
    accuracy = 2 # accuracy of the finite difference scheme (2-points, 4-points, etc.)
    d_dx_class = findiff.FinDiff(1, step_size, order, acc=accuracy)
    d_dy_class = findiff.FinDiff(0, step_size, order, acc=accuracy)
    D_x = d_dx_class.matrix((num_pix_x, num_pix_y))
    D_y = d_dy_class.matrix((num_pix_x, num_pix_y))  # sparse matrices
    
    # join the source and potential derivatives operators
    # through minus their 'scalar' product (Eq. A6 from Koopmans 2005)
    DsD = - D_s_x.dot(D_x) - D_s_y.dot(D_y)

    # we also return the gradient of the source after being ray-traced to the data grid
    return DsD, grad_s_x, grad_s_y
