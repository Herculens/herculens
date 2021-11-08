import copy
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import morphology
from scipy import sparse, ndimage
import findiff

from herculens.LightModel.light_model import LightModel
from herculens.LensImage.lens_image import LensImage
from herculens.Util import param_util
from herculens.Util.jax_util import BicubicInterpolator as Interpolator


def mask_from_source_area(lens_image, parameters):
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


def mask_from_lensed_source(lens_image, parameters=None, source_model=None,
                            threshold=0.1, smoothing=0):
    if parameters is None and source_model is None:
        raise ValueError("You must provide an the source model "
                         "if no parameters are provided.")
    if parameters is not None:
        kwargs_param = parameters.current_values(as_kwargs=True)
        source_model = lens_image.source_surface_brightness(kwargs_param['kwargs_source'], de_lensed=True, unconvolved=True)
    source_model = np.array(source_model)
    if smoothing > 0:
        source_model = ndimage.gaussian_filter(source_model, sigma=smoothing)
    binary_source = source_model / source_model.max()
    binary_source[binary_source < threshold] = 0.
    binary_source[binary_source >= threshold] = 1.
    grid = copy.deepcopy(lens_image.Grid)
    grid.remove_model_grid('source')
    lens_image_pixel = LensImage(grid, lens_image.PSF, 
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


def build_bilinear_interpol_matrix(x_grid_1d_in, y_grid_1d_in, x_grid_1d_out, 
                                   y_grid_1d_out, warning=True):
    """
    Only works with square input and output grids.
    Author: austinpeel, originally for the package `slitronomy`.
    """
    # Standardize inputs for vectorization
    x_grid_1d_out = np.atleast_1d(x_grid_1d_out)
    y_grid_1d_out = np.atleast_1d(y_grid_1d_out)
    assert len(x_grid_1d_out) == len(y_grid_1d_out), "Input arrays must be the same size."
    num_pix_out = len(x_grid_1d_out)
    
    # Compute bin edges so that (x_coord, y_coord) lie at the grid centers
    num_pix = int(np.sqrt(x_grid_1d_in.size))
    delta_pix = np.abs(x_grid_1d_in[0] - x_grid_1d_in[1])
    half_pix = delta_pix / 2.

    x_coord = x_grid_1d_in[:num_pix]
    x_dir = -1 if x_coord[0] > x_coord[-1] else 1  # Handle x-axis inversion
    x_lower = x_coord[0] - x_dir * half_pix
    x_upper = x_coord[-1] + x_dir * half_pix
    xbins = np.linspace(x_lower, x_upper, num_pix + 1)

    y_coord = y_grid_1d_in[::num_pix]
    y_dir = -1 if y_coord[0] > y_coord[-1] else 1  # Handle y-axis inversion
    y_lower = y_coord[0] - y_dir * half_pix
    y_upper = y_coord[-1] + y_dir * half_pix
    ybins = np.linspace(y_lower, y_upper, num_pix + 1)

    # Keep only coordinates that fall within the output grid
    x_min, x_max = [x_lower, x_upper][::x_dir]
    y_min, y_max = [y_lower, y_upper][::y_dir]
    selection = ((x_grid_1d_out > x_min) & (x_grid_1d_out < x_max) &
                 (y_grid_1d_out > y_min) & (y_grid_1d_out < y_max))
    if np.any(1 - selection.astype(int)):
        x_grid_1d_out = x_grid_1d_out[selection]
        y_grid_1d_out = y_grid_1d_out[selection]
        num_pix_out = len(x_grid_1d_out)

    # Find the (1D) output pixel that (x_grid_1d_out, y_grid_1d_out) falls in
    index_x = np.digitize(x_grid_1d_out, xbins) - 1
    index_y = np.digitize(y_grid_1d_out, ybins) - 1
    index_1 = index_x + index_y * num_pix

    # Compute distances between input and output grid points
    dx = x_grid_1d_out - x_grid_1d_in[index_1]
    dy = y_grid_1d_out - y_grid_1d_in[index_1]

    # Find the three other nearest pixels (may end up out of bounds)
    index_2 = index_1 + x_dir * np.sign(dx).astype(int)
    index_3 = index_1 + y_dir * np.sign(dy).astype(int) * num_pix
    index_4 = index_2 + y_dir * np.sign(dy).astype(int) * num_pix

    # Treat these index arrays as four sets stacked vertically
    # Prepare to mask out out-of-bounds pixels as well as repeats
    # The former is important for the csr_matrix to be generated correctly
    max_index = x_grid_1d_in.size - 1  # Upper index bound
    mask = np.ones((4, num_pix_out), dtype=bool)  # Mask for the coordinates

    # Mask out any neighboring pixels that end up out of bounds
    mask[1, np.where((index_2 < 0) | (index_2 > max_index))[0]] = False
    mask[2, np.where((index_3 < 0) | (index_3 > max_index))[0]] = False
    mask[3, np.where((index_4 < 0) | (index_4 > max_index))[0]] = False

    # Mask any repeated pixels (2 or 3x) arising from unlucky grid alignment
    # zero_dx = list(np.where(dx == 0)[0])
    # zero_dy = list(np.where(dy == 0)[0])
    # unique, counts = np.unique(zero_dx + zero_dy, return_counts=True)
    # repeat_row = [ii + 1 for c in counts for ii in range(0, 3, 3 - c)]
    # repeat_col = [u for (u, c) in zip(unique, counts) for _ in range(c + 1)]
    #mask[(repeat_row, repeat_col)] = False  # TODO: this leads to strange lines

    # Generate 2D indices of non-zero elements for the sparse matrix
    row = np.tile(np.nonzero(selection)[0], (4, 1))
    col = np.array([index_1, index_2, index_3, index_4])

    # Compute bilinear weights like in Treu & Koopmans (2004)
    col[~mask] = 0  # Avoid accessing values out of bounds
    dist_x = (np.tile(x_grid_1d_out, (4, 1)) - x_grid_1d_in[col]) / delta_pix
    dist_y = (np.tile(y_grid_1d_out, (4, 1)) - y_grid_1d_in[col]) / delta_pix
    weight = (1 - np.abs(dist_x)) * (1 - np.abs(dist_y))

    # Make sure the weights are properly normalized
    # This step is only necessary where the mask has excluded source pixels
    norm = np.expand_dims(np.sum(weight, axis=0, where=mask), 0)
    weight = weight / norm

    if warning:
        if np.any(weight[mask] < 0):
            num_neg = np.sum((weight[mask] < 0).astype(int))
            print("Warning : {} weights are negative.".format(num_neg))

    indices, weights = (row[mask], col[mask]), weight[mask]

    dense_shape = (x_grid_1d_out.size, x_grid_1d_in.size)
    interpol_matrix = sparse.csr_matrix((weights, indices), shape=dense_shape)
    interpol_norm = np.squeeze(np.maximum(1, interpol_matrix.sum(axis=0)).A)
    return interpol_matrix, interpol_norm


def build_DsD_matrix(smooth_lens_image, smooth_kwargs_params, hybrid_lens_image=None):
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
