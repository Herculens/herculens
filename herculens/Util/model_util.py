# Utility functions
# 
# Copyright (c) 2021, herculens developers and contributors

__author__ = 'aymgal'


import copy
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import morphology
from scipy import ndimage
from skimage import measure

from herculens.LensImage.lensing_operator import LensingOperator


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
                            threshold=0.1, smoothing=0, kwargs_numerics=None):
    # imports are here to avoid issues with circular imports
    from herculens.LensImage.lens_image import LensImage
    from herculens.LightModel.light_model import LightModel

    if parameters is None and source_model is None:
        raise ValueError("You must provide an the source model "
                         "if no parameters are provided.")
    if parameters is not None:
        kwargs_param = parameters.current_values(as_kwargs=True)
        source_model = lens_image.source_surface_brightness(kwargs_param['kwargs_source'], 
                                                            de_lensed=True, unconvolved=True)
    source_model = np.array(source_model)
    if smoothing > 0:
        source_model = ndimage.gaussian_filter(source_model, sigma=smoothing)
    binary_source = source_model / source_model.max()
    binary_source[binary_source < threshold] = 0.
    binary_source[binary_source >= threshold] = 1.
    grid = copy.deepcopy(lens_image.Grid)
    lens_image_pixel = LensImage(grid, lens_image.PSF, 
                                 noise_class=lens_image.Noise,
                                 lens_mass_model_class=lens_image.MassModel,
                                 source_model_class=LightModel(['PIXELATED']),
                                 lens_light_model_class=lens_image.LensLightModel,
                                 kwargs_numerics=kwargs_numerics)
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


def pixelated_region_from_sersic(kwargs_sersic, force_square=False, use_major_axis=False,
                                 min_width=1.0, min_height=1.0, scaling=1.0):
    # imports are here to avoid issues with circular imports
    from herculens.Util import param_util

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
    if force_square is True:
        width = max(width, height)
        height = width
    # the following dict is consistent with arguments of PixelGrid.create_model_grid()
    kwargs_pixelated_grid = {
        'grid_center': (float(c_x), float(c_y)),
        'grid_shape': (float(width), float(height)),
    }
    return kwargs_pixelated_grid


def pixelated_region_from_arc_mask(arc_mask, image_grid, mass_model, mass_params):
    # We first design a hypothetical source plane with sufficiently high resolution
    nx, ny = image_grid.num_pixel_axes
    nx_src, ny_src = (nx//3, ny//3)
    high_res_source_grid = image_grid.create_model_grid(pixel_scale_factor=0.5, grid_shape=(nx_src, ny_src))
    
    # Then build the lensing operator based on the image/source grids and mass model
    lensing_op = LensingOperator(mass_model, image_grid, high_res_source_grid)
    lensing_op.compute_mapping(kwargs_lens=mass_params)
    
    # De-lens the arc mask
    arc_mask_source = np.array(lensing_op.image2source_2d(arc_mask))
    arc_mask_source = np.where(arc_mask_source < 0.1, 0., 1.)  # only contains 0 and 1 now

    # Find the minimum bounding box that encloses the mask in source plane (square here)
    # - get the extrema 
    rows = np.max(arc_mask_source, axis=0)
    print(rows)
    row_low, row_high = np.argmax(rows), np.argmax(rows[::-1])
    print("R", row_low, row_high)
    cols = np.max(arc_mask_source, axis=1)
    print(cols)
    col_low, col_high = np.argmax(cols), np.argmax(cols[::-1])
    print("C", col_low, col_high)
    # - get the pixel coordinates corresponding to these extrema
    x_low, y_low = high_res_source_grid.map_pix2coord(row_low, col_low)
    x_high, y_high = high_res_source_grid.map_pix2coord(row_high, col_high)
    print(x_low, y_low)
    print(x_high, y_high)

    # Get the grid parameters
    new_grid_shape = (abs(x_high - x_low), abs(y_high - y_low))
    new_grid_center = (0.5*(x_low+x_high), 0.5*(y_low+y_high))
    kwargs_pixelated_grid = {
        'grid_center': new_grid_center,
        'grid_shape': new_grid_shape,
    }

    # Create the new, reduced source plane grid
    return kwargs_pixelated_grid


def estimate_model_covariance(lens_image, parameters, samples, return_cross_covariance=False):
    model_samples = []
    for sample in samples:
        kwargs_sample = parameters.args2kwargs(sample)
        model_sample = lens_image.model(**kwargs_sample)
        model_map_shape = model_sample.shape
        model_samples.append(model_sample.flatten())
    model_samples = np.array(model_samples)
    
    # variance map (as a 2D image)
    model_var_map = np.var(model_samples, axis=0).reshape(*model_map_shape)
    
    # full (auto) covariance matrix from the model samples
    model_cov = np.cov(model_samples, rowvar=False)

    # computation of the cross-covariance matrix from model and data samples
    if return_cross_covariance:
        data_mean_proxy = lens_image.model(**parameters.best_fit_values(as_kwargs=True))   # best-fit model as a proxy for the unknown data 'mean'
        data_var = lens_image.Noise.C_D_model(data_mean_proxy, force_recompute=True)
        data_mean_proxy = data_mean_proxy.flatten()
        data_cov = np.diag(data_var.flatten())
        data_samples = draw_samples_from_covariance(data_mean_proxy, data_cov, num_samples=len(model_samples))
        data_vector  = np.mean(data_samples - data_samples.mean(axis=0), axis=0)
        model_vector = np.mean(model_samples - model_samples.mean(axis=0), axis=0)
        data_model_cross_cov = np.outer(data_vector, model_vector)
        return model_var_map, model_cov, data_model_cross_cov
    else:
        return model_var_map, model_cov


def draw_samples_from_covariance(mean, covariance, num_samples=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.multivariate_normal(mean, covariance, size=num_samples)
    return samples


def critical_curves(lens_image, kwargs_lens, return_lens_centers=False):
    # TODO: for some reason, using the numerics grid does lead to proper pix2coord conversions 
    # grid = lens_image.ImageNumerics.grid_class
    # x_grid_img, y_grid_img = grid.coordinates_evaluate

    # evaluate the total magnification
    grid = lens_image.Grid
    x_grid_img, y_grid_img = grid.pixel_coordinates
    mag_tot = lens_image.MassModel.magnification(x_grid_img, y_grid_img, kwargs_lens)
    # mag_tot = util.array2image(mag_tot)

    # invert and find contours corresponding to infite magnification
    inv_mag_tot = 1. / np.array(mag_tot)
    contours = measure.find_contours(inv_mag_tot, 0.)

    # convert to model coordinates
    curves = []
    for i, contour in enumerate(contours):
        curve_x, curve_y = grid.map_pix2coord(contour[:, 1], contour[:, 0])
        curves.append((np.array(curve_x), np.array(curve_y)))

    # can also returns the lens components centroids for convenience
    if return_lens_centers:
        cxs, cys = [], []
        for kw in kwargs_lens:
            if 'center_x' in kw:
                cxs.append(kw['center_x'])
                cys.append(kw['center_y'])
        return curves, (np.array(cxs), np.array(cys))
    
    return curves

def shear_deflection_field(lens_image, kwargs_lens, num_pixels=20):
    shear_type = 'SHEAR_GAMMA_PSI'
    try:
        shear_idx = lens_image.MassModel.profile_type_list.index(shear_type)
    except ValueError:
        shear_type = 'SHEAR'
        try:
            shear_idx = lens_image.MassModel.profile_type_list.index(shear_type)
        except ValueError:
            return None
    if shear_type == 'SHEAR_GAMMA_PSI':
        # imports are here to avoid issues with circular imports
        from herculens.Util import param_util
        gamma1, gamma2 = param_util.shear_polar2cartesian(kwargs_lens[shear_idx]['phi_ext'],
                                                          kwargs_lens[shear_idx]['gamma_ext'])
    else:
        gamma1, gamma2 = kwargs_lens[shear_idx]['gamma1'], kwargs_lens[shear_idx]['gamma2']
    grid = lens_image.Grid.create_model_grid(num_pixels=num_pixels)
    x_grid_img, y_grid_img = grid.pixel_coordinates
    alpha_x, alpha_y = lens_image.MassModel.alpha(x_grid_img, y_grid_img, kwargs_lens, 
                                                  k=shear_idx)
    return (np.array(x_grid_img), np.array(y_grid_img), 
            gamma1, gamma2, np.array(alpha_x), np.array(alpha_y))
